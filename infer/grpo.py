import argparse
import json
import os
import random
import re

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image, ImageFile
from qwen_vl_utils import process_vision_info
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.answers import extract_grpo_answer
from utils.data import TASK_CONFIGS
from utils.io import save_examples, save_results
from utils.metrics import compute_metrics, make_serializable
from utils.prompts import get_system_prompt

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for GRPO-trained figurative language models.")
    parser.add_argument("--task", required=True, choices=["sarcasm", "humor", "metaphor", "offensive"])
    parser.add_argument("--model-path", required=True, help="Path to trained GRPO model")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-prefix", default="grpo_evaluation")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def load_dataset_for_task(task: str, split: str, multimet_path: str, num_samples: int):
    cfg = TASK_CONFIGS[task]
    if task == "metaphor":
        ds = load_from_disk(multimet_path)[split]
    elif task in ("humor", "offensive"):
        ds = load_dataset("Ahren09/MMSoc_Memotion", split=split)
    else:
        ds = load_dataset("coderchen01/MMSD2.0", "mmsd-clean", split=split)

    if split == "train" or (num_samples and num_samples < len(ds)):
        size = num_samples or 5000
        random.seed(42)
        indices = random.sample(range(len(ds)), min(size, len(ds)))
        ds = ds.select(indices)

    ds = ds.map(lambda ex: {
        "image": _preprocess(ex["image"]),
        "text": ex[cfg.text_field],
        "label": cfg.label_fn(ex[cfg.raw_label_field]),
    })
    return ds


def _preprocess(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((448, 448), Image.Resampling.LANCZOS)


def generate_with_reasoning(model, processor, system_prompt: str, text: str, image: Image.Image) -> str:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=500)
    return processor.decode(output_ids[0], skip_special_tokens=True)


def main():
    args = parse_args()
    task = args.task
    system_prompt = get_system_prompt(task, "grpo")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True, padding_side="left")

    print(f"Loading {task} {args.split} dataset...")
    ds = load_dataset_for_task(task, args.split, args.multimet_path, args.num_samples)
    print(f"Evaluating on {len(ds)} examples...")

    predictions, labels, responses = [], [], []

    for idx, example in enumerate(tqdm(ds, desc="Evaluating")):
        generated_text = generate_with_reasoning(model, processor, system_prompt, example["text"], example["image"])
        prediction = extract_grpo_answer(generated_text, task)
        if prediction is not None:
            predictions.append(prediction)
            labels.append(example["label"])
            responses.append(generated_text)
        else:
            print(f"Warning: Could not extract prediction from example {idx}")

        if (idx + 1) % 100 == 0:
            print(f"Progress: {idx + 1}/{len(ds)} — valid: {len(predictions)}")

    print(f"\nValid predictions: {len(predictions)} / {len(ds)}")

    metrics = compute_metrics(labels, predictions)
    metrics["total_examples"] = len(ds)
    metrics["valid_predictions"] = len(predictions)

    print("\n----- EVALUATION RESULTS -----")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    model_tag = args.model_path.replace("/", "_")
    results_file = f"{args.output_prefix}_{model_tag}_{args.split}_results.json"
    examples_file = f"{args.output_prefix}_{model_tag}_{args.split}_examples.json"

    save_results(metrics, results_file)
    save_examples(
        [{"index": i, "true_label": l, "predicted_label": p, "reasoning": r}
         for i, (l, p, r) in enumerate(zip(labels, predictions, responses))],
        examples_file,
    )
    print(f"Results saved to: {results_file}")
    print(f"Examples saved to: {examples_file}")


if __name__ == "__main__":
    main()
