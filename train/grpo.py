import argparse
import random
import re
import os
from collections import Counter

import torch
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFile
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

TASK_LABEL_MAP = {
    "sarcasm":   {1: "sarcastic",    0: "not sarcastic"},
    "humor":     {1: "humorous",     0: "not humorous"},
    "metaphor":  {1: "metaphorical", 0: "not metaphorical"},
    "offensive": {1: "offensive",    0: "not offensive"},
}

VALID_ANSWERS = [
    "sarcastic", "not sarcastic",
    "humorous", "not humorous",
    "metaphorical", "not metaphorical",
    "offensive", "not offensive",
]

MAX_PROMPT_LENGTHS = {
    "sarcasm": 2048, "combined": 2048,
    "humor": 6000, "metaphor": 6000, "offensive": 6000,
}


def preprocess_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((448, 448), Image.Resampling.LANCZOS)


def filter_image(example: dict) -> bool:
    w, h = example["image"].size
    return w > 56 and h > 56


def get_system_prompt(task: str) -> str:
    from utils.prompts import get_system_prompt as _gsp
    return _gsp(task, "grpo")


def load_single_task_dataset(task: str, multimet_path: str):
    if task == "metaphor":
        return load_from_disk(multimet_path)["train"]
    elif task in ("humor", "offensive"):
        ds = load_dataset("Ahren09/MMSoc_Memotion", split="train")
        return ds.map(lambda ex: {"image": preprocess_image(ex["image"])})
    elif task == "sarcasm":
        ds = load_dataset("coderchen01/MMSD2.0", "mmsd-clean", split="train")
        ds = ds.filter(filter_image)
        return ds.map(lambda ex: {"image": preprocess_image(ex["image"])})
    raise ValueError(f"Unknown task: {task}")


def get_label(task: str, example: dict) -> int:
    if task == "sarcasm":
        return int(example["label"])
    elif task == "humor":
        return 0 if example["humor"] == "not_funny" else 1
    elif task == "metaphor":
        return int(example["metaphor"])
    elif task == "offensive":
        return 0 if example["offensive"] == "not_offensive" else 1
    raise ValueError(f"Unknown task: {task}")


def make_conversation_single(task: str, example: dict, processor) -> dict:
    system_prompt = get_system_prompt(task)
    text_field = "text_corrected" if task in ("humor", "offensive") else "text"
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": example[text_field]},
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {"prompt": prompt, "image": example["image"], "label": get_label(task, example)}


def make_conversation_combined(task: str, example: dict, processor) -> dict:
    system_prompt = get_system_prompt(task)
    text_field = example.get("text_field", "text")
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": example.get(text_field, example.get("text", ""))},
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {"prompt": prompt, "image": preprocess_image(example["image"]), "label": get_label(task, example), "task": task}


def build_combined_dataset(sample_size: int, multimet_path: str, processor) -> Dataset:
    tasks_data = []

    metaphor_ds = load_from_disk(multimet_path)["train"]
    metaphor_items = [dict(ex, task="metaphor", text_field="text") for ex in metaphor_ds]
    if len(metaphor_items) > sample_size:
        random.seed(42)
        metaphor_items = random.sample(metaphor_items, sample_size)
    tasks_data.extend(metaphor_items)

    memotion_ds = load_dataset("Ahren09/MMSoc_Memotion", split="train")
    memotion_list = list(memotion_ds)
    if len(memotion_list) > sample_size:
        random.seed(42)
        shared_indices = random.sample(range(len(memotion_list)), sample_size)
    else:
        shared_indices = list(range(len(memotion_list)))
    for i in shared_indices:
        tasks_data.append(dict(memotion_list[i], task="humor", text_field="text_corrected"))
    for i in shared_indices:
        tasks_data.append(dict(memotion_list[i], task="offensive", text_field="text_corrected"))

    sarcasm_ds = load_dataset("coderchen01/MMSD2.0", "mmsd-clean", split="train")
    sarcasm_ds = sarcasm_ds.filter(filter_image)
    sarcasm_list = list(sarcasm_ds)
    if len(sarcasm_list) > sample_size:
        random.seed(42)
        sarcasm_list = random.sample(sarcasm_list, sample_size)
    tasks_data.extend([dict(ex, task="sarcasm", text_field="text") for ex in sarcasm_list])

    all_keys = set().union(*[set(d.keys()) for d in tasks_data])
    tasks_data = [{k: d.get(k) for k in all_keys} for d in tasks_data]

    ds = Dataset.from_list(tasks_data)
    ds = ds.shuffle(seed=42)

    processed = []
    for ex in ds:
        processed.append(make_conversation_combined(ex["task"], ex, processor))
    return Dataset.from_list(processed)


def format_reward(completions, **kwargs):
    strict_pattern = r"^<think>\s*\n?.*?\n?\s*</think>\s*\n<answer>\s*\n?.*?\n?\s*</answer>$"
    rewards = []
    for completion in completions:
        reward = 0.0
        if re.match(strict_pattern, completion, re.DOTALL | re.MULTILINE):
            think_m = re.search(r"<think>\s*(.*?)\s*</think>", completion, re.DOTALL | re.IGNORECASE)
            answer_m = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, re.DOTALL | re.IGNORECASE)
            if think_m and answer_m:
                think_content = think_m.group(1)
                answer_content = answer_m.group(1).strip().lower()
                has_steps = all(
                    bool(re.search(rf"step\s*{n}\s*:", think_content, re.IGNORECASE))
                    for n in range(1, 5)
                )
                valid_answer = any(ans in answer_content for ans in VALID_ANSWERS)
                if has_steps and valid_answer:
                    reward = 1.0
        rewards.append(reward)
    return rewards


def build_accuracy_reward(task: str, class_weights: dict):
    label_map = TASK_LABEL_MAP.get(task)

    def accuracy_reward_single(completions, **kwargs):
        rewards = []
        labels = kwargs.get("label", [])
        for completion, label in zip(completions, labels):
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, re.DOTALL | re.IGNORECASE)
            if not m:
                rewards.append(0.0)
                continue
            prediction = re.sub(r"[^\w\s]", "", m.group(1).strip().lower())
            true_label = re.sub(r"[^\w\s]", "", label_map[int(label)])
            base = 1.0 if prediction == true_label else 0.0
            weight = class_weights.get(int(label), 1.0) if class_weights else 1.0
            rewards.append(base * weight)
        return rewards

    def accuracy_reward_combined(completions, **kwargs):
        rewards = []
        labels = kwargs.get("label", [])
        tasks = kwargs.get("task", [])
        for completion, label, t in zip(completions, labels, tasks):
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, re.DOTALL | re.IGNORECASE)
            if not m:
                rewards.append(0.0)
                continue
            prediction = re.sub(r"[^\w\s]", "", m.group(1).strip().lower())
            true_label = re.sub(r"[^\w\s]", "", TASK_LABEL_MAP[t][int(label)])
            rewards.append(1.0 if prediction == true_label else 0.0)
        return rewards

    return accuracy_reward_combined if task == "combined" else accuracy_reward_single


def compute_class_weights(dataset) -> dict:
    label_counts = Counter(dataset["label"])
    num_classes = len(label_counts)
    total = sum(label_counts.values())
    raw = {c: total / (num_classes * cnt) for c, cnt in label_counts.items()}
    majority = max(raw.items(), key=lambda x: label_counts[x[0]])[0]
    return {c: w / raw[majority] for c, w in raw.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--task", required=True, choices=["sarcasm", "humor", "metaphor", "offensive", "combined"])
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sft-style", default=None, help="SFT init tag for output dir naming")
    parser.add_argument("--weighted-reward", action="store_true", help="Use class-weighted accuracy reward")
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task
    max_prompt_length = args.max_prompt_length or MAX_PROMPT_LENGTHS.get(task, 2048)
    output_dir = f"Qwen2.5-VL-3B-Instruct-{task}-gen4-" + (f"sft-{args.sft_style}" if args.sft_style else "without-sft")

    print(f"GRPO Training — task: {task}, model: {args.model_id}, sft_style: {args.sft_style}")

    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True, padding_side="left")

    if task == "combined":
        sample_size = args.sample_size or 5000
        train_dataset = build_combined_dataset(sample_size, args.multimet_path, processor)
        class_weights = {}
    else:
        raw_dataset = load_single_task_dataset(task, args.multimet_path)
        if args.sample_size and len(raw_dataset) > args.sample_size:
            random.seed(42)
            indices = random.sample(range(len(raw_dataset)), args.sample_size)
            raw_dataset = raw_dataset.select(indices)
        train_dataset = raw_dataset.map(lambda ex: make_conversation_single(task, ex, processor))

        class_weights = compute_class_weights(train_dataset) if args.weighted_reward else {}
        if args.weighted_reward:
            print("Class weights:", class_weights)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    accuracy_reward = build_accuracy_reward(task, class_weights)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        remove_unused_columns=False,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        bf16=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=10,
        max_completion_length=1024,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        report_to="tensorboard",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
