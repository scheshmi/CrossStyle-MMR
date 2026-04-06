import argparse
import base64
import io
import json
import os

from datasets import load_dataset, load_from_disk
from PIL import Image, ImageFile
from tqdm import tqdm
from vllm import LLM, SamplingParams

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "./hf"))
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

SYSTEM_PROMPTS = {
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


def load_model(model_id: str) -> LLM:
    return LLM(
        model=model_id,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=2,
        max_model_len=20000,
        max_num_seqs=4,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        disable_custom_all_reduce=True,
    )


def generate_batch(model: LLM, task: str, images: list, texts: list) -> list:
    system_prompt = SYSTEM_PROMPTS[task]
    prompts = []
    for image, text in zip(images, texts):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        prompts.append([{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": f"{system_prompt} {text}"},
            ],
        }])
    outputs = model.chat(prompts, sampling_params=SamplingParams(temperature=0.1, max_tokens=512, top_p=0.9))
    return [o.outputs[0].text.strip() for o in outputs]


def load_split(task: str, split: str, multimet_path: str, num_samples: int):
    if task == "metaphor":
        ds = load_from_disk(multimet_path)[split]
    elif task in ("humor", "offensive"):
        load_str = f"{split}[0:{num_samples}]" if num_samples else split
        ds = load_dataset("Ahren09/MMSoc_Memotion", split=load_str, streaming=(num_samples is None))
    elif task == "sarcasm":
        load_str = f"{split}[0:{num_samples}]" if num_samples else split
        ds = load_dataset("coderchen01/MMSD2.0", "mmsd-clean", split=load_str, streaming=(num_samples is None))
    else:
        raise ValueError(f"Unknown task: {task}")
    return ds


def build_record(task: str, example: dict, idx: int, reasoning: str, split: str) -> dict:
    text_field = "text_corrected" if task in ("humor", "offensive") else "text"
    record = {"text": example.get(text_field, ""), "reasoning_trace": reasoning}
    if task == "sarcasm":
        record["id"] = str(example.get("id", f"sample_{idx}"))
        record["label"] = int(example.get("label", 0))
    elif task == "humor":
        record["id"] = f"sample_{idx}"
        record["humor_original"] = example.get("humor", "")
        record["offensive_original"] = example.get("offensive", "")
    elif task == "offensive":
        record["id"] = f"sample_{idx}"
        record["offensive_original"] = example.get("offensive", "")
        record["humor_original"] = example.get("humor", "")
    elif task == "metaphor":
        record["id"] = f"{split}_{idx}"
        record["split"] = split
        record["metaphor_gt"] = example.get("metaphor", 0)
    return record


def process_split(model: LLM, task: str, split: str, multimet_path: str, num_samples: int,
                  output_file: str, batch_size: int) -> None:
    ds = load_split(task, split, multimet_path, num_samples)
    processed = 0
    batch_data = []

    with open(output_file, "w") as f:
        for idx, example in enumerate(tqdm(iter(ds), desc=f"{task}/{split}")):
            image_raw = example.get("image")
            text = example.get("text_corrected" if task in ("humor", "offensive") else "text", "")
            if not image_raw or not text:
                continue
            if hasattr(image_raw, "convert"):
                image = image_raw
            elif isinstance(image_raw, dict) and image_raw.get("bytes"):
                image = Image.open(io.BytesIO(image_raw["bytes"]))
            else:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")
            batch_data.append({"image": image, "example": example, "idx": idx})

            if len(batch_data) == batch_size:
                reasonings = generate_batch(model, task, [b["image"] for b in batch_data], [b["example"].get("text_corrected" if task in ("humor", "offensive") else "text", "") for b in batch_data])
                for item, reasoning in zip(batch_data, reasonings):
                    f.write(json.dumps(build_record(task, item["example"], item["idx"], reasoning, split)) + "\n")
                    f.flush()
                    processed += 1
                batch_data = []

        if batch_data:
            reasonings = generate_batch(model, task, [b["image"] for b in batch_data], [b["example"].get("text_corrected" if task in ("humor", "offensive") else "text", "") for b in batch_data])
            for item, reasoning in zip(batch_data, reasonings):
                f.write(json.dumps(build_record(task, item["example"], item["idx"], reasoning, split)) + "\n")
                f.flush()
                processed += 1

    print(f"Wrote {processed} records to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning traces for a figurative language detection task.")
    parser.add_argument("--task", required=True, choices=["humor", "offensive", "metaphor", "sarcasm"])
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-90B-Vision-Instruct")
    parser.add_argument("--split", default="train", choices=["train", "test", "both"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=None, help="Output .jsonl path (auto-generated if omitted)")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--multimet-path", default="final_multimet_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_id)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        output = args.output or f"reasoning_traces_{split}_{args.task}.jsonl"
        process_split(model, args.task, split, args.multimet_path, args.num_samples, output, args.batch_size)

    print("Done.")


if __name__ == "__main__":
    main()
