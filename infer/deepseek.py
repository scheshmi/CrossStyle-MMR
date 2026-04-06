"""
Inference for DeepSeek-VL2 on sarcasm detection (MMSD2.0).
Requires the DeepSeek-VL2 repo to be cloned and importable.
Pass --deepseek-repo-path to add it to sys.path if needed.
"""
import argparse
import gc
import json
import os
import re
import sys

import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True

_utils_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _utils_path)

from utils.answers import extract_step5_answer
from utils.data import load_dataset_for_task
from utils.io import save_examples, save_results
from utils.metrics import compute_metrics, make_serializable

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = (
    "Analyze the provided image and caption to determine if the pair is sarcastic or not sarcastic. "
    "Provide your reasoning in the following format:\n"
    "Step 1: What the image shows: [Detailed description of the image content]\n"
    "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
    "Step 3: Detecting mismatch: [Explain if there is a mismatch or congruence between the image and caption, and why]\n"
    "Step 4: Inference of intent: [Conclude whether the intent is sarcastic or not based on the mismatch/congruence]\n"
    "Step 5: Final answer: [Provide your final answer in the form of sarcastic or not sarcastic for image-caption pair]\n\n"
    "Caption:"
)


def load_model(model_id: str) -> tuple:
    from deepseek_vl2.models import DeepseekVLV2Processor

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model, tokenizer, processor


def unload_model(model, tokenizer, processor) -> None:
    del model, tokenizer, processor
    gc.collect()
    torch.cuda.empty_cache()


def run_inference(model, tokenizer, processor, text: str, image: Image.Image, device: str = "cuda") -> str:
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{SYSTEM_PROMPT} {text}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    try:
        inputs = processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            system_prompt="",
        ).to(device)
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        with torch.no_grad():
            outputs = model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
        response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        del inputs, inputs_embeds, outputs
        torch.cuda.empty_cache()
        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return ""


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-VL2 inference for sarcasm detection.")
    parser.add_argument("--model-id", default="deepseek-ai/deepseek-vl2")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=1, help="Effective batch size (recommended: 1)")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-prefix", default="deepseek_vl2_evaluation")
    parser.add_argument("--deepseek-repo-path", default=None, help="Path to DeepSeek-VL2 repo (added to sys.path)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.deepseek_repo_path:
        sys.path.insert(0, args.deepseek_repo_path)

    ds = load_dataset_for_task("sarcasm", split=args.split)
    if args.sample_size and args.sample_size < len(ds):
        ds = ds.select(range(args.sample_size))
    ds = ds.map(lambda ex, idx: {**ex, "id": ex.get("id", f"sarcasm_{idx}")}, with_indices=True)

    model, tokenizer, processor = load_model(args.model_id)

    print(f"Evaluating on {len(ds)} examples...")
    predictions, labels, ids, responses = [], [], [], []

    for idx, example in enumerate(tqdm(ds, desc="Evaluating")):
        response = run_inference(model, tokenizer, processor, example["text"], example["image"])
        pred = extract_step5_answer(response, "sarcasm")
        if pred is not None:
            predictions.append(pred)
            labels.append(example["label"])
            ids.append(example.get("id", idx))
            responses.append(response)
        else:
            print(f"Warning: Could not extract prediction from example {idx}")

    print(f"\nValid predictions: {len(predictions)} / {len(ds)}")
    unload_model(model, tokenizer, processor)

    metrics = compute_metrics(labels, predictions) if predictions else {}
    metrics["total_examples"] = len(ds)
    metrics["valid_predictions"] = len(predictions)

    print("\n----- RESULTS -----")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    results_file = f"{args.output_prefix}_results.json"
    examples_file = f"{args.output_prefix}_examples.json"

    save_results({k: make_serializable(v) for k, v in metrics.items()}, results_file)
    save_examples(
        [{"id": make_serializable(id_), "true_label": make_serializable(l),
          "predicted_label": make_serializable(p), "reasoning": r}
         for id_, l, p, r in zip(ids, labels, predictions, responses)],
        examples_file,
    )
    print(f"Results saved to: {results_file}")
    print(f"Examples saved to: {examples_file}")


if __name__ == "__main__":
    main()
