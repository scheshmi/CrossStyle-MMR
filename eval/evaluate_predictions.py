import argparse
import json
import re
import sys
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data import TASK_CONFIGS


def extract_answer_from_reasoning(text: str, task: str) -> int | None:
    cfg = TASK_CONFIGS[task]

    # Try "Final answer:" line first
    m = re.search(r"Final answer:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        answer = m.group(1).strip().lower()
        if cfg.neg_keyword in answer:
            return 0
        if cfg.pos_keyword in answer and cfg.neg_keyword not in answer:
            return 1

    # Fallback to keyword scan
    if cfg.neg_keyword in text.lower():
        return 0
    if cfg.pos_keyword in text.lower():
        return 1
    return None


def load_and_evaluate(jsonl_path: str, task: str) -> None:
    cfg = TASK_CONFIGS[task]
    predictions, ground_truth = [], []
    failed = 0

    reasoning_key = "generated_reasoning"
    label_key = f"{task}_binary"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                reasoning = data.get(reasoning_key) or data.get("reasoning_trace", "")
                label_raw = data.get(label_key)
                if label_raw is None:
                    label_raw = data.get("label")

                pred = extract_answer_from_reasoning(reasoning, task)
                if pred is not None and label_raw is not None:
                    predictions.append(pred)
                    ground_truth.append(int(label_raw))
                else:
                    failed += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error on line {line_num}: {e}")

    print(f"Valid predictions: {len(predictions)} | Failed: {failed}")

    if not predictions:
        print("No valid predictions found.")
        return

    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average="binary", zero_division=0.0)
    recall = recall_score(ground_truth, predictions, average="binary", zero_division=0.0)
    f1 = f1_score(ground_truth, predictions, average="binary", zero_division=0.0)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples:  {len(predictions)}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")

    pos_label = cfg.pos_keyword.title()
    neg_label = cfg.neg_keyword.title()
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=[neg_label, pos_label]))

    cm = confusion_matrix(ground_truth, predictions)
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              0    1")
    print(f"Actual    0  {cm[0, 0]:4d} {cm[0, 1]:4d}")
    print(f"          1  {cm[1, 0]:4d} {cm[1, 1]:4d}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reasoning trace predictions from a JSONL file.")
    parser.add_argument("--input", required=True, help="Path to input .jsonl file")
    parser.add_argument("--task", required=True, choices=["sarcasm", "humor", "metaphor", "offensive"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_and_evaluate(args.input, args.task)
