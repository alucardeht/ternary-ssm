# Training Efficiency — Maximum Performance on Consumer Hardware (February 2026)

## Hardware Baseline
- RTX 5070: 12GB GDDR7 VRAM
- Intel Core Ultra 5 225F: CPU only (NPU cannot train)
- 32GB DDR5 RAM

## The VRAM Budget Problem

### Why Full Fine-Tuning Doesn't Fit

```
Full fine-tuning 7B (FP16):
  Weights:              16 GB (2 bytes × 8B params)
  Gradients:            16 GB (copy of weights)
  Optimizer (Adam):     32 GB (2 states × 16 bytes each)
  Activations:          8-16 GB
  TOTAL:                72-80 GB → IMPOSSIBLE on 12GB
```

The optimizer is the killer. Adam stores momentum + variance = 2 extra values per parameter = 2x the model size just for optimizer state.

---

## Techniques Ranked by Impact

### 1. QLoRA (Baseline — Proven)

Freeze base model in 4-bit, train small LoRA adapters in FP16.

```
Weights (4-bit nf4):    4.5 GB
LoRA adapters (FP16):   0.5 GB  ← only this is trained
Gradients (LoRA only):  1.0 GB
Optimizer (8-bit):      1.5 GB
Activations:            3.0 GB (with gradient checkpointing)
CUDA overhead:          1.0 GB
TOTAL:                  11.5 GB → FITS (tight)
```

- Quality: 80-90% of full fine-tuning
- Speed: fast (1-2 it/s on RTX 5070)
- Framework: Unsloth (2-5x faster, 70% less VRAM than HF)
- Status: Production, proven, easy

### 2. AdaLomo + 4-bit Quantization (Best Quality in 12GB)

Fuses gradient computation with parameter update. Zero extra optimizer states.

```
Weights (4-bit):       4.5 GB
Gradients:             0 GB (fused with backward pass)
Optimizer:             ~0.5 GB (1 state per group, not per param)
Activations:           3.0 GB
Overhead:              ~1.0 GB
TOTAL:                 ~8-9 GB → FITS COMFORTABLY
```

- Quality: MATCHES AdamW (not an approximation — proven equivalent)
- Speed: moderate
- Framework: lomo-optim package, integrates with Transformers
- Status: Production (ACL 2024), well-documented
- Note: Original LOMO is inferior — use AdaLomo specifically

### 3. Q-GaLore (Experimental — Highest Potential)

Projects gradients to low-rank space via SVD. Enables FULL-RANK parameter updates (not adapter-based like LoRA) with low memory.

```
Q-GaLore 7B training:
  ~12-16 GB (borderline for 12GB, may work with aggressive settings)

  Outperforms QLoRA by up to 5.19 MMLU points
  50% less memory than standard GaLore
  Full-rank updates (not low-rank adapters)
```

- Quality: 95%+ of full fine-tuning (better than QLoRA)
- Speed: moderate
- Framework: https://github.com/VITA-Group/Q-GaLore
- Status: Experimental (July 2024 paper)
- Risk: may exceed 12GB VRAM

### 4. DoRA (Weight-Decomposed LoRA)

Decomposes LoRA update into magnitude and direction. Same VRAM as QLoRA, better quality.

```
Same memory as QLoRA: ~11.5 GB
Quality: 92-95% of full FT (vs 80-90% for QLoRA)
```

- Status: Experimental (2025), worth trying if framework supports it

### 5. Gradient Checkpointing (MANDATORY — Use With Everything)

Trades compute for memory. Recomputes activations during backward pass instead of storing them.

```
VRAM savings: ~40% on activation memory
Speed cost: ~20-30% slower training
```

Advanced variants:
- Selective Activation Checkpointing (SAC): only recompute cheap ops
- EAGLE (2025): closed-form gradients for linear layers, 13-18% faster than full recompute
- Dynamic checkpointing: adapts based on runtime memory availability

Status: Production, built into PyTorch. NON-NEGOTIABLE for 12GB training.

---

## What Does NOT Work on 12GB

| Technique | Why Not |
|-----------|---------|
| FSDP CPU offload | 32GB RAM insufficient (needs ~128GB for 7B) |
| Standard GaLore | Needs 24GB for 7B |
| ReLoRA | Doesn't outperform LoRA for fine-tuning (only good for pre-training) |
| CPU/NPU training | 50-100x slower than GPU |
| Full FP16 fine-tuning | 72-80GB required |
| 3B+ pre-training | Exceeds 12GB even with all optimizations |

---

## NPU and CPU for Training: Verdict

### Intel NPU (13 TOPS)
- NO backward pass support (inference-only hardware by design)
- No gradient computation units
- No framework supports NPU training
- VERDICT: Completely useless for training

### Intel CPU (Core Ultra 5 225F)
- AMX support unclear for desktop Arrow Lake (typically Xeon feature)
- CPU training is 10-100x slower than GPU
- Only useful for FSDP optimizer state offloading (limited by 32GB RAM)
- VERDICT: Not practical for training

### Conclusion
Use RTX 5070 exclusively for training. Period.

---

## Pre-Training From Scratch

### Feasible on RTX 5070?

| Model Size | Technique | VRAM | Time (10B tokens) | Feasible? |
|------------|-----------|------|--------------------|-----------|
| 500M | Q-GaLore + checkpoint | ~8-10 GB | ~20-30 days | YES |
| 1B | Q-GaLore + checkpoint | ~12-15 GB | ~40-60 days | MAYBE |
| 3B | Any | >16 GB | months | NO |

### BitNet Pre-Training
If Model v2 (BitNet + SSM) works, pre-training becomes cheaper:
- Weights are 12x smaller (ternary)
- Gradient memory proportionally reduced
- Could potentially pre-train 3B BitNet on 12GB
- UNVALIDATED — no one has tried this combination

### Frameworks
- LitGPT: 20+ architectures, pre-training support, clean code
- nanoGPT: educational, <1B models

---

## Continual Pre-Training vs Fine-Tuning

| | Continual Pre-Training | Fine-Tuning |
|-|----------------------|-------------|
| Purpose | Inject domain knowledge | Adapt to tasks/style |
| Data | Billions of tokens (raw text) | Thousands of examples (Q&A pairs) |
| VRAM (with QLoRA) | ~12 GB (fits) | ~11.5 GB (fits) |
| Duration | Weeks-months | Hours |
| Knowledge injection | YES | NO (only style/format) |

Best practice: Continual pre-training FIRST (domain knowledge) → then fine-tuning (task adaptation).

Both fit in 12GB with QLoRA + Unsloth.

---

## Recommended Stack

### 7B Fine-Tuning (Start Here)
```
Unsloth + QLoRA + Gradient Checkpointing
  VRAM: 10-11 GB
  Quality: 95%+ with Unsloth optimizations
  Speed: 2-5x faster than HF standard
  Setup: pip install unsloth
```

### 7B Maximum Quality Fine-Tuning
```
AdaLomo + 4-bit quantization + Gradient Checkpointing
  VRAM: 8-9 GB
  Quality: Matches full AdamW fine-tuning
  Speed: moderate
  Setup: pip install lomo-optim
```

### Domain Knowledge Injection
```
Unsloth + QLoRA (rank 64) + Gradient Checkpointing
  VRAM: 6-7 GB
  Data: 1-10B tokens of domain text
  Duration: weeks on RTX 5070
```

### Small Model Pre-Training
```
LitGPT + Q-GaLore + Gradient Checkpointing
  Models: 500M-1B
  VRAM: 8-12 GB
  Duration: 20-60 days
  Use case: educational, experimentation
```

---

## Memory Optimization Checklist

1. `per_device_train_batch_size=1` (non-negotiable on 12GB)
2. `gradient_accumulation_steps=4-8` (effective batch size)
3. `gradient_checkpointing=True` (MANDATORY)
4. `bf16=True` (RTX 5070 supports bfloat16)
5. `optim="paged_adamw_8bit"` (for QLoRA) or AdaLomo
6. `max_seq_length=512` (start conservative, increase to 1024 if fits)
7. Close Ollama before training (shared VRAM between Windows and WSL2)
8. `torch.cuda.empty_cache()` after model loading

## If OOM (priority order)
1. Reduce max_seq_length (512 → 256)
2. Enable flash_attention_2
3. Reduce LoRA rank (16 → 8)
4. Reduce gradient_accumulation_steps
5. NEVER disable gradient_checkpointing

---

*Last updated: February 2026*
