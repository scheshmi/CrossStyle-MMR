# Reasoning Beyond Literal: Cross-style Multimodal Reasoning for Figurative Language Understanding

This directory contains the official implementation of the paper:

> **Reasoning Beyond Literal: Cross-style Multimodal Reasoning for Figurative Language Understanding**  (EACL 2026)
> Seyyed Saeid Cheshmi, Hahnemann Ortiz, James Mooney, Dongyeop Kang
> University of Minnesota  
> [Paper](https://aclanthology.org/2026.findings-eacl.311.pdf)

---
# Abstract
Vision–language models (VLMs) have demonstrated strong reasoning abilities in literal multimodal tasks such as visual mathematics and science question answering. However, figurative language—such as sarcasm, humor, and metaphor—remains a significant challenge, as it conveys intent and emotion through subtle incongruities between expressed and intended meanings. In multimodal settings, accompanying images can amplify or invert textual meaning, demanding models that reason across modalities and account for subjectivity. We propose a three-step framework for developing efficient multimodal reasoning models that can (i) interpret multimodal figurative language, (ii) provide transparent reasoning traces, and (iii) generalize across multiple figurative styles. Experiments across four styles show that (1) incorporating reasoning traces substantially improves multimodal figurative understanding, (2) reasoning learned in one style can transfer to others—especially between related styles like sarcasm and humor, and (3) training jointly across styles yields a generalized reasoning VLM that outperforms much larger open- and closed-source models.Our findings show that lightweight VLMs with verifiable reasoning achieve robust cross-style generalization while providing inspectable reasoning traces for multimodal tasks.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

### Step 1 — Generate reasoning traces (SFT warm-up)

```bash
python train/generate_reasoning.py \
  --task humor \
  --model-id meta-llama/Llama-3.2-90B-Vision-Instruct \
  --split train \
  --output reasoning_traces_humor.jsonl
```

### Step 2 — SFT on distilled reasoning traces

Per-task CoT SFT:
```bash
python train/sft.py --task sarcasm   --mode cot --reasoning-file reasoning_traces_sarcasm.jsonl
python train/sft.py --task humor     --mode cot --reasoning-file reasoning_traces_humor.jsonl
python train/sft.py --task metaphor  --mode cot --reasoning-file reasoning_traces_metaphor.jsonl
python train/sft.py --task offensive --mode cot --reasoning-file reasoning_traces_offensive.jsonl
```

Combined CoT SFT (all 4 tasks, 5K samples each):
```bash
python train/sft_combined.py \
  --sarcasm-reasoning-file reasoning_traces_sarcasm.jsonl \
  --humor-reasoning-file reasoning_traces_humor.jsonl \
  --metaphor-reasoning-file reasoning_traces_metaphor.jsonl \
  --offensive-reasoning-file reasoning_traces_offensive.jsonl
```

### Step 3 — GRPO fine-tuning

Single task:
```bash
python train/grpo.py --task sarcasm --sft-style sarcasm
python train/grpo.py --task metaphor --sft-style combined
```

Multi-task (combined):
```bash
python train/grpo.py --task combined --sft-style combined
```

---

## Inference

### GRPO-trained model

```bash
python infer/grpo.py \
  --task humor \
  --model-path ./Qwen2.5-VL-3B-Instruct-humor-grpo \
  --output-prefix grpo_humor
```

### CoT SFT model

```bash
python infer/sft_cot.py \
  --task humor \
  --model-path ./qwen2.5-vl-3b-humor-cot-merged
```

### Binary SFT model

```bash
python infer/binary.py \
  --task humor \
  --model-path ./qwen2.5-vl-3b-humor-binary-merged
```

### vLLM baselines

```bash
python infer/vllm_baseline.py \
  --model-id meta-llama/Llama-3.2-90B-Vision-Instruct \
  --task sarcasm \
  --use-cot
```

### Gemini

```bash
python infer/gemini.py --task sarcasm --use-cot
```

---


## Citation

```bibtex
@inproceedings{cheshmi-etal-2026-reasoning,
    title = "Reasoning Beyond Literal: Cross-style Multimodal Reasoning for Figurative Language Understanding",
    author = "Cheshmi, Seyyed Saeid  and Ortiz, Hahnemann  and Mooney, James  and Kang, Dongyeop",
    editor = "Demberg, Vera  and Inui, Kentaro  and Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.311/",
    doi = "10.18653/v1/2026.findings-eacl.311",
    pages = "5942--5956",
    ISBN = "979-8-89176-386-9",
}
```
