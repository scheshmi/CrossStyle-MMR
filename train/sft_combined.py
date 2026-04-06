import argparse
import json
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

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

COT_PROMPTS = {
    "metaphor": (
        "You are an expert at detecting metaphors in images and text. When given an image and text, analyze whether the "
        "content uses metaphorical language or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Metaphor cues: [Explain if there are figurative expressions, symbolic comparisons, or non-literal meanings "
        "that connect the caption and the image]\n"
        "Step 4: Interpretation: [Discuss what abstract idea, concept, or meaning the metaphor might be conveying]\n"
        "Step 5: Final answer: [Strictly answer with one of the two options: \"metaphorical\" or \"not metaphorical\" and don't add any other text]\n\nCaption:"
    ),
    "humor": (
        "You are an expert at detecting humor in images and text. When given an image and text, analyze whether the "
        "content is humorous or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Humor cues: [Explain if there are elements such as exaggeration, wordplay, absurdity, or incongruity "
        "between the image and caption that make the content humorous]\n"
        "Step 4: Inference of intent: [Conclude whether the intent is humorous or not based on the cues]\n"
        "Step 5: Final answer: [Strictly answer with one of the two options: \"humorous\" or \"not humorous\" and don't add any other text]\n\nCaption:"
    ),
    "offensive": (
        "You are an expert at detecting offensive content in images and text. When given an image and text, analyze whether "
        "the content is offensive or not. Provide your reasoning process in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Offense cues: [Explain if there are elements such as hate speech, slurs, derogatory language, demeaning "
        "stereotypes, harassment, or explicit insults that make the content offensive]\n"
        "Step 4: Context and intent: [Discuss whether the content was likely meant to harm, insult, or demean someone]\n"
        "Step 5: Final answer: [Strictly answer with one of the two options: \"offensive\" or \"not offensive\" and don't add any other text]\n\nCaption:"
    ),
    "sarcasm": (
        "Analyze the provided image and caption to determine if the pair is sarcastic or not sarcastic. "
        "Provide your reasoning in the following format:\n"
        "Step 1: What the image shows: [Detailed description of the image content]\n"
        "Step 2: What the caption says: [Quote or paraphrase the caption]\n"
        "Step 3: Detecting mismatch: [Explain if there is a mismatch or congruence between the image and caption, and why]\n"
        "Step 4: Inference of intent: [Conclude whether the intent is sarcastic or not based on the mismatch/congruence]\n"
        "Step 5: Final answer: [Provide your final answer in the form of sarcastic or not sarcastic for image-caption pair]\n\nCaption:"
    ),
}


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_conversation(task: str, sample: dict, reasoning_trace: str, processor) -> dict:
    system_prompt = COT_PROMPTS[task]
    text_field = "text_corrected" if task in ("humor", "offensive") else "text"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{system_prompt} {sample[text_field]}"},
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


def create_metaphor_dataset(reasoning_file: str, multimet_path: str, sample_size: int, processor) -> list:
    reasoning_data = {r["id"]: r for r in load_jsonl(reasoning_file)}
    ds = load_from_disk(multimet_path)["train"]
    matched = []
    for idx, item in enumerate(ds):
        rid = f"train_{idx}"
        if rid in reasoning_data:
            matched.append(make_conversation("metaphor", item, reasoning_data[rid]["reasoning_trace"], processor))
    if len(matched) > sample_size:
        random.seed(42)
        matched = random.sample(matched, sample_size)
    return matched


def create_memotion_dataset(task: str, reasoning_file: str, sample_size: int, sampled_indices, processor):
    reasoning_data = load_jsonl(reasoning_file)
    reasoning_map = {r["id"]: r for r in reasoning_data}
    ds = load_dataset("Ahren09/MMSoc_Memotion", split="train")
    matched = []
    for idx, item in enumerate(ds):
        rid = f"sample_{idx}"
        if rid in reasoning_map:
            rt = reasoning_map[rid].get("generated_reasoning") or reasoning_map[rid].get("reasoning_trace", "")
            matched.append({"item": item, "reasoning_trace": rt, "original_idx": idx})

    if sampled_indices is None and len(matched) > sample_size:
        random.seed(42)
        sampled_indices = random.sample(range(len(matched)), sample_size)

    if sampled_indices:
        matched = [matched[i] for i in sampled_indices if i < len(matched)]

    conversations = [make_conversation(task, m["item"], m["reasoning_trace"], processor) for m in matched]
    return conversations, sampled_indices


def create_sarcasm_dataset(reasoning_file: str, sample_size: int, processor) -> list:
    reasoning_map = {str(r["id"]): r for r in load_jsonl(reasoning_file)}
    from datasets import load_dataset as _ld
    ds = _ld("coderchen01/MMSD2.0", "mmsd-clean", split="train")
    ds = ds.filter(lambda ex: ex["image"].size[0] > 56 and ex["image"].size[1] > 56)
    matched = []
    for item in ds:
        rid = str(item["id"])
        if rid in reasoning_map:
            rt = reasoning_map[rid].get("generated_reasoning") or reasoning_map[rid].get("reasoning_trace", "")
            matched.append(make_conversation("sarcasm", item, rt, processor))
    if len(matched) > sample_size:
        random.seed(42)
        matched = random.sample(matched, sample_size)
    return matched


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task reasoning SFT on combined figurative language dataset.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs-qwen-3b-reasoning-combined-sft")
    parser.add_argument("--merged-dir", default="qwen2.5-vl-3b-combined-reasoning-merged")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    parser.add_argument("--metaphor-reasoning-file", default="reasoning_traces_train_metaphor.jsonl")
    parser.add_argument("--humor-reasoning-file", default="output_reasoning_memotion_humor.jsonl")
    parser.add_argument("--offensive-reasoning-file", default="output_reasoning_memotion_offensive.jsonl")
    parser.add_argument("--sarcasm-reasoning-file", default="output_reasoning.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

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

    print("Building combined dataset (5K per task)...")
    metaphor_data = create_metaphor_dataset(args.metaphor_reasoning_file, args.multimet_path, args.sample_size, processor)
    humor_data, shared_indices = create_memotion_dataset("humor", args.humor_reasoning_file, args.sample_size, None, processor)
    offensive_data, _ = create_memotion_dataset("offensive", args.offensive_reasoning_file, args.sample_size, shared_indices, processor)
    sarcasm_data = create_sarcasm_dataset(args.sarcasm_reasoning_file, args.sample_size, processor)

    all_conversations = metaphor_data + humor_data + offensive_data + sarcasm_data
    random.seed(42)
    random.shuffle(all_conversations)
    final_dataset = Dataset.from_list(all_conversations)
    print(f"Combined dataset: {len(final_dataset)} examples")

    training_args = SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        logging_steps=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=args.output_dir,
        report_to="tensorboard",
        save_strategy="epoch",
        remove_unused_columns=False,
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=final_dataset,
        args=training_args,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)

    hf_write_token = os.getenv("HF_TOKEN_WRITE")
    if hf_write_token:
        login(token=hf_write_token)

    model = trainer.model.merge_and_unload()
    model.save_pretrained(args.merged_dir)
    tokenizer.save_pretrained(args.merged_dir)
    processor.save_pretrained(args.merged_dir)
    print(f"Merged model saved to: {args.merged_dir}")


if __name__ == "__main__":
    main()
