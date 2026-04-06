import json
from pathlib import Path

from utils.metrics import make_serializable


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_results(metrics: dict, path: str) -> None:
    Path(path).write_text(
        json.dumps({k: make_serializable(v) for k, v in metrics.items()}, indent=2)
    )


def save_examples(examples: list, path: str) -> None:
    with open(path, "w") as f:
        json.dump(
            [{k: make_serializable(v) for k, v in ex.items()} for ex in examples],
            f,
            indent=2,
        )
