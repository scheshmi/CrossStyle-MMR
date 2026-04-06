import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

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


def load_model(model_id: str) -> tuple:
    print(f"Loading model: {model_id}")
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=4,
        max_model_len=85000,
        max_num_seqs=4,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return llm, tokenizer, processor


def unload_model(llm, tokenizer, processor) -> None:
    del llm, tokenizer, processor
    gc.collect()


def build_prompt(sample: dict, system_prompt: str, processor, model_id: str) -> dict:
    text = f"{system_prompt} {sample['text']}"
    if "phi-4" in model_id.lower():
        messages = [{"role": "user", "content": text}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image"},
        ]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt_text, "multi_modal_data": {"image": sample["image"]}}


def process_batch(llm, processor, batch: list, system_prompt: str, task: str, model_id: str) -> tuple:
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    batch_inputs = [build_prompt(s, system_prompt, processor, model_id) for s in batch]
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    predictions, responses = [], []
    for output in outputs:
        response = output.outputs[0].text
        prediction = extract_step5_answer(response, task)
        predictions.append(prediction)
        responses.append(response)
    return predictions, responses


def create_batches(dataset, batch_size: int) -> list:
    batches = []
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def evaluate(llm, tokenizer, processor, dataset, model_id: str, task: str, batch_size: int, use_cot: bool) -> dict:
    mode = "cot" if use_cot else "binary"
    system_prompt = get_system_prompt(task, mode)
    batches = create_batches(dataset, batch_size)

    print(f"Evaluating {model_id} on {task} (CoT={use_cot}) — {len(dataset)} samples in {len(batches)} batches...")

    predictions, labels, ids, responses = [], [], [], []

    for batch in tqdm(batches):
        batch_preds, batch_resps = process_batch(llm, processor, batch, system_prompt, task, model_id)
        for sample, pred, resp in zip(batch, batch_preds, batch_resps):
            if pred is not None:
                predictions.append(pred)
                labels.append(sample["label"])
                ids.append(sample.get("id", ""))
                responses.append(resp)

    metrics = compute_metrics(labels, predictions) if predictions else {k: 0.0 for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")}
    metrics["total_examples"] = len(dataset)
    metrics["valid_predictions"] = len(predictions)

    print("\n----- RESULTS -----")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return {"metrics": metrics, "predictions": predictions, "labels": labels, "id": ids, "responses": responses}


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM baseline evaluation for figurative language detection.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-90B-Vision-Instruct")
    parser.add_argument("--task", default="sarcasm", choices=["sarcasm", "humor", "metaphor", "offensive"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset_for_task(args.task, split=args.split, multimet_disk_path=args.multimet_path)
    if args.sample_size and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))

    dataset = dataset.map(lambda ex, idx: {**ex, "id": ex.get("id", f"{args.task}_{idx}")}, with_indices=True)

    llm, tokenizer, processor = load_model(args.model_id)
    results = evaluate(llm, tokenizer, processor, dataset, args.model_id, args.task, args.batch_size, args.use_cot)
    unload_model(llm, tokenizer, processor)

    model_name = args.model_id.split("/")[-1].lower().replace("-", "_")
    cot_suffix = "_cot" if args.use_cot else "_zeroshot"
    results_file = f"{model_name}_{args.task}{cot_suffix}_results.json"
    examples_file = f"{model_name}_{args.task}{cot_suffix}_examples.json"

    save_results({k: make_serializable(v) for k, v in results["metrics"].items()}, results_file)
    save_examples(
        [{"id": make_serializable(id_), "true_label": make_serializable(l),
          "predicted_label": make_serializable(p), "reasoning": r}
         for id_, l, p, r in zip(results["id"], results["labels"], results["predictions"], results["responses"])],
        examples_file,
    )
    print(f"\nResults saved to: {results_file}")
    print(f"Examples saved to: {examples_file}")


if __name__ == "__main__":
    main()
