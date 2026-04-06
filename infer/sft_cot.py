import argparse
import os
import sys

import torch
from PIL import ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.answers import extract_step5_answer
from utils.data import load_dataset_for_task
from utils.io import save_examples, save_results
from utils.metrics import compute_metrics, make_serializable
from utils.prompts import get_system_prompt

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def load_model(model_path: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, processor


def run_inference(task: str, model, tokenizer, processor, sample: dict) -> str:
    system_prompt = get_system_prompt(task, "cot")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{system_prompt} {sample['text']}"},
                {"type": "image", "image": sample["image"]},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample["image"]], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for CoT SFT figurative language models.")
    parser.add_argument("--task", required=True, choices=["sarcasm", "humor", "metaphor", "offensive"])
    parser.add_argument("--model-path", required=True, help="Path to merged CoT SFT model")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-prefix", default="sft_cot_evaluation")
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task

    model, tokenizer, processor = load_model(args.model_path)

    ds = load_dataset_for_task(task, split=args.split, multimet_disk_path=args.multimet_path)
    if args.num_samples and args.num_samples < len(ds):
        ds = ds.select(range(args.num_samples))

    print(f"Evaluating {task} ({args.split}) on {len(ds)} examples...")

    predictions, labels, responses = [], [], []

    for idx, example in enumerate(tqdm(ds, desc="Evaluating")):
        response = run_inference(task, model, tokenizer, processor, example)
        pred = extract_step5_answer(response, task)
        if pred is not None:
            predictions.append(pred)
            labels.append(example["label"])
            responses.append(response)
        else:
            print(f"Warning: Could not extract prediction from example {idx}")

    print(f"\nValid predictions: {len(predictions)} / {len(ds)}")

    metrics = compute_metrics(labels, predictions) if predictions else {}
    metrics["total_examples"] = len(ds)
    metrics["valid_predictions"] = len(predictions)

    print("\n----- RESULTS -----")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    model_tag = args.model_path.rstrip("/").split("/")[-1]
    results_file = f"{args.output_prefix}_{task}_{model_tag}_results.json"
    examples_file = f"{args.output_prefix}_{task}_{model_tag}_examples.json"

    save_results({k: make_serializable(v) for k, v in metrics.items()}, results_file)
    save_examples(
        [{"index": i, "true_label": make_serializable(l),
          "predicted_label": make_serializable(p), "reasoning": r}
         for i, (l, p, r) in enumerate(zip(labels, predictions, responses))],
        examples_file,
    )
    print(f"Results saved to: {results_file}")
    print(f"Examples saved to: {examples_file}")


if __name__ == "__main__":
    main()
