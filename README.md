# Multimodal Figurative Language Detection

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
python train/grpo.py --task sarcasm
python train/grpo.py --task humor   --weighted-reward --sft-style humor
python train/grpo.py --task metaphor --weighted-reward --sft-style combined
python train/grpo.py --task offensive --weighted-reward --sft-style combined
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

### vLLM baseline (LLaVA, Phi-4, LLaMA, Qwen-32B, etc.)

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
@article{,
}
```
