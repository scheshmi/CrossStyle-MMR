import re

from utils.data import TASK_CONFIGS


def extract_grpo_answer(response: str, task: str):
    cfg = TASK_CONFIGS[task]
    matches = re.findall(r"<answer>[^<]*?</answer>", response, re.IGNORECASE | re.DOTALL)
    if not matches:
        return None
    answer_text = re.sub(r"</?answer>", "", matches[-1], flags=re.IGNORECASE).strip().lower()
    if answer_text == cfg.neg_keyword:
        return 0
    if answer_text == cfg.pos_keyword:
        return 1
    return None


def extract_step5_answer(response: str, task: str):
    cfg = TASK_CONFIGS[task]
    text = response.split("assistant\n")[-1] if "assistant\n" in response else response

    pattern = (
        rf"Step 5:.*?(?:final answer|answer)?:?\s*"
        rf"({re.escape(cfg.neg_keyword)}|{re.escape(cfg.pos_keyword)})"
    )
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        return 0 if cfg.neg_keyword in m.group(1).lower() else 1

    if cfg.neg_keyword in text.lower():
        return 0
    if cfg.pos_keyword in text.lower():
        return 1
    return None
