import argparse
import os
import sys
import time

import google.generativeai as genai
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

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


def setup_model(api_key: str = None) -> genai.GenerativeModel:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass --api-key.")
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")


def process_batch(model, batch: list, system_prompt: str, task: str, rate_limit_delay: float) -> tuple:
    predictions, responses, times = [], [], []
    for sample in batch:
        try:
            prompt = f"{system_prompt} {sample['text']}"
            start = time.time()
            response = model.generate_content([prompt, sample["image"]])
            elapsed = time.time() - start
            response_text = response.text
            prediction = extract_step5_answer(response_text, task)
        except Exception as e:
            print(f"Error processing sample: {e}")
            response_text = f"Error: {e}"
            prediction = None
            elapsed = 0
        predictions.append(prediction)
        responses.append(response_text)
        times.append(elapsed)
        time.sleep(rate_limit_delay)
    return predictions, responses, times


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


def evaluate(model, dataset, task: str, batch_size: int, use_cot: bool, rate_limit_delay: float) -> dict:
    mode = "cot" if use_cot else "binary"
    system_prompt = get_system_prompt(task, mode)
    batches = create_batches(dataset, batch_size)

    print(f"Evaluating Gemini on {task} (CoT={use_cot}) — {len(dataset)} samples...")

    predictions, labels, ids, responses, inference_times = [], [], [], [], []

    for batch in tqdm(batches):
        batch_preds, batch_resps, batch_times = process_batch(model, batch, system_prompt, task, rate_limit_delay)
        for sample, pred, resp, t in zip(batch, batch_preds, batch_resps, batch_times):
            if pred is not None:
                predictions.append(pred)
                labels.append(sample["label"])
                ids.append(sample.get("id", ""))
                responses.append(resp)
                inference_times.append(t)

    metrics = compute_metrics(labels, predictions) if predictions else {k: 0.0 for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")}
    metrics["total_examples"] = len(dataset)
    metrics["valid_predictions"] = len(predictions)
    if inference_times:
        t = np.array(inference_times)
        metrics.update({
            "mean_inference_time": float(np.mean(t)),
            "total_inference_time": float(np.sum(t)),
        })

    print("\n----- RESULTS -----")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return {"metrics": metrics, "predictions": predictions, "labels": labels, "id": ids, "responses": responses}


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini API evaluation for figurative language detection.")
    parser.add_argument("--api-key", default=None, help="Gemini API key (default: $GEMINI_API_KEY env var)")
    parser.add_argument("--task", default="sarcasm", choices=["sarcasm", "humor", "metaphor", "offensive"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-prefix", default="gemini_evaluation")
    parser.add_argument("--rate-limit-delay", type=float, default=1.0)
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset_for_task(args.task, split=args.split, multimet_disk_path=args.multimet_path)
    if args.sample_size and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))

    dataset = dataset.map(lambda ex, idx: {**ex, "id": ex.get("id", f"{args.task}_{idx}")}, with_indices=True)

    model = setup_model(args.api_key)
    results = evaluate(model, dataset, args.task, args.batch_size, args.use_cot, args.rate_limit_delay)

    cot_suffix = "_cot" if args.use_cot else "_zeroshot"
    results_file = f"{args.output_prefix}_{args.task}{cot_suffix}_results.json"
    examples_file = f"{args.output_prefix}_{args.task}{cot_suffix}_examples.json"

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
