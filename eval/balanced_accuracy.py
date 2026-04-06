import json
import sys
from pathlib import Path


def balanced_accuracy_from_counts(tp, fp, tn, fn):
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr), tpr, tnr


def main(path: str):
    data = json.loads(Path(path).read_text())
    preds = data["predictions"]
    labels = data["labels"]
    if len(preds) != len(labels):
        raise ValueError("Predictions and labels length mismatch")
    tp = fp = tn = fn = 0
    for p, y in zip(preds, labels):
        if y == 1 and p == 1:
            tp += 1
        elif y == 1 and p == 0:
            fn += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
        else:
            raise ValueError(f"Unexpected label/pred pair ({p},{y})")
    bal_acc, tpr, tnr = balanced_accuracy_from_counts(tp, fp, tn, fn)
    print(f"Samples: {len(preds)}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Recall(pos/TPR)={tpr:.12f}")
    print(f"Recall(neg/TNR)={tnr:.12f}")
    print(f"Balanced Accuracy={bal_acc:.12f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python balanced_accuracy.py <results.json>")
        sys.exit(1)
    main(sys.argv[1])
