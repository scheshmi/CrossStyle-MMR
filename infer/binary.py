import argparse
import json
import os
import sys

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data import TASK_CONFIGS
from utils.io import save_results
from utils.metrics import compute_metrics

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

BINARY_PROMPTS = {
    "humor":     "Based on the given image and the caption, classify if the image and caption is humorous or not. \ncaption: ",
    "metaphor":  "Based on the given image and the caption, classify if the image and caption contain metaphor or not. \ncaption: ",
    "offensive": "Based on the given image and the caption, classify if the image and caption is offensive or not. \ncaption: ",
}


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
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer, processor


def create_conversation(task: str, sample: dict) -> list:
    cfg = TASK_CONFIGS[task]
    system_message = BINARY_PROMPTS[task]
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{system_message}{sample[cfg.text_field]}"},
            {"type": "image", "image": sample["image"]},
        ],
    }]


def predict_label(task: str, response: str) -> int:
    cfg = TASK_CONFIGS[task]
    lower = response.lower()
    if "0" in response or cfg.neg_keyword in lower:
        return 0
    return 1


def predict_batch(task: str, model, tokenizer, processor, batch: list) -> tuple:
    conversations = [create_conversation(task, s) for s in batch]
    texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
    images = [s["image"] for s in batch]

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [predict_label(task, r) for r in responses], responses


def load_eval_dataset(task: str, split: str, multimet_path: str):
    cfg = TASK_CONFIGS[task]
    if task == "metaphor":
        return load_from_disk(multimet_path)[split]
    return load_dataset(cfg.dataset_id, split=split)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for binary SFT figurative language models.")
    parser.add_argument("--task", required=True, choices=["humor", "metaphor", "offensive"])
    parser.add_argument("--model-path", required=True, help="Path to binary-merged model")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None, help="Output JSON file (default: {task}_binary_inference_results.json)")
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task
    output_file = args.output or f"{task}_binary_inference_results.json"
    cfg = TASK_CONFIGS[task]

    model, tokenizer, processor = load_model(args.model_path)
    ds = load_eval_dataset(task, args.split, args.multimet_path)

    all_predictions, all_labels, all_responses = [], [], []

    for i in tqdm(range(0, len(ds), args.batch_size)):
        batch = [ds[j] for j in range(i, min(i + args.batch_size, len(ds)))]
        predictions, responses = predict_batch(task, model, tokenizer, processor, batch)
        labels = [cfg.label_fn(s[cfg.raw_label_field]) for s in batch]
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_responses.extend(responses)

    metrics = compute_metrics(all_labels, all_predictions)
    print("\n----- RESULTS -----")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    outputs = [{"prediction": p, "label": l, "response": r}
               for p, l, r in zip(all_predictions, all_labels, all_responses)]
    with open(output_file, "w") as f:
        json.dump({"metrics": metrics, "outputs": outputs}, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
