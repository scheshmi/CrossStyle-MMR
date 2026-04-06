"""
Per-task SFT in two modes:
  binary: train directly on label text (no reasoning file needed)
  cot:    train on distilled reasoning traces (requires --reasoning-file)
"""
import argparse
import os
import random

import torch
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import login
from peft import LoraConfig
from PIL import ImageFile
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data import TASK_CONFIGS, load_dataset_for_task
from utils.io import load_jsonl
from utils.prompts import get_system_prompt

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

BINARY_PROMPTS = {
    "humor": "Based on the given image and the caption, classify if the image and caption is humorous or not. \ncaption: ",
    "metaphor": "Based on the given image and the caption, classify if the image and caption contain metaphor or not. \ncaption: ",
    "offensive": "Based on the given image and the caption, classify if the image and caption is offensive or not. \ncaption: ",
    "sarcasm": "Based on the given image and the caption, classify if the image and caption is sarcastic or not sarcastic. \ncaption: ",
}


def _reasoning_map(reasoning_file: str) -> dict:
    return {str(r["id"]): r for r in load_jsonl(reasoning_file)}


def _get_reasoning_trace(record: dict) -> str:
    return record.get("reasoning_trace") or record.get("generated_reasoning", "")


def _id_for_task(task: str, item: dict, idx: int) -> str:
    if task == "sarcasm":
        return str(item["id"])
    elif task == "metaphor":
        return f"train_{idx}"
    else:
        return f"sample_{idx}"


def make_binary_conversation(task: str, sample: dict, processor) -> dict:
    cfg = TASK_CONFIGS[task]
    label_text = cfg.pos_keyword if sample["label"] == 1 else cfg.neg_keyword
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{BINARY_PROMPTS[task]}{sample['text']}"},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"The predicted label is {label_text} or {sample['label']}"}],
        },
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    return {"text": text, "images": [sample["image"]]}


def make_cot_conversation(task: str, sample: dict, reasoning_trace: str, processor) -> dict:
    system_prompt = get_system_prompt(task, "cot")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{system_prompt} {sample['text']}"},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": reasoning_trace}],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "images": [sample["image"]]}


def build_binary_dataset(task: str, multimet_path: str, processor) -> list:
    ds = load_dataset_for_task(task, split="train", multimet_disk_path=multimet_path)
    return [make_binary_conversation(task, s, processor) for s in ds]


def build_cot_dataset(task: str, reasoning_file: str, multimet_path: str, processor) -> list:
    rmap = _reasoning_map(reasoning_file)
    if task == "metaphor":
        ds = load_from_disk(multimet_path)["train"]
    elif task == "sarcasm":
        ds = load_dataset("coderchen01/MMSD2.0", "mmsd-clean", split="train")
        ds = ds.filter(lambda ex: ex["image"].size[0] > 56 and ex["image"].size[1] > 56)
    else:
        ds = load_dataset("Ahren09/MMSoc_Memotion", split="train")

    matched = []
    for idx, item in enumerate(ds):
        rid = _id_for_task(task, item, idx)
        if rid not in rmap:
            continue
        rt = _get_reasoning_trace(rmap[rid])
        cfg = TASK_CONFIGS[task]
        text = item[cfg.text_field]
        label = cfg.label_fn(item[cfg.raw_label_field])
        normalized = {"image": item["image"], "text": text, "label": label}
        matched.append(make_cot_conversation(task, normalized, rt, processor))
    return matched


def parse_args():
    parser = argparse.ArgumentParser(description="Per-task SFT for figurative language detection.")
    parser.add_argument("--task", required=True, choices=["sarcasm", "humor", "metaphor", "offensive"])
    parser.add_argument("--mode", default="cot", choices=["binary", "cot"])
    parser.add_argument("--reasoning-file", default=None, help="JSONL of reasoning traces (required for --mode cot)")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "cot" and not args.reasoning_file:
        raise ValueError("--reasoning-file is required when --mode cot")

    mode_suffix = "cot" if args.mode == "cot" else "binary"
    output_dir = args.output_dir or f"outputs-qwen-3b-{args.task}-sft-{mode_suffix}"
    merged_dir = args.merged_dir or f"qwen2.5-vl-3b-{args.task}-{mode_suffix}-merged"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    print(f"Building {args.mode} dataset for {args.task}...")
    if args.mode == "binary":
        conversations = build_binary_dataset(args.task, args.multimet_path, processor)
    else:
        conversations = build_cot_dataset(args.task, args.reasoning_file, args.multimet_path, processor)

    random.seed(42)
    random.shuffle(conversations)
    dataset = Dataset.from_list(conversations)
    print(f"Dataset size: {len(dataset)}")

    training_args = SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=output_dir,
        report_to="tensorboard",
        save_strategy="epoch",
        remove_unused_columns=False,
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)

    hf_write_token = os.getenv("HF_TOKEN_WRITE")
    if hf_write_token:
        login(token=hf_write_token)

    model = trainer.model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
