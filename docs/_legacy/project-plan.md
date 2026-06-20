# Local LLM Project - Plan & Roadmap

## Overview

**Goal:** Run and fine-tune 7-8B parameter LLMs locally, serve via API on the network.

**Hardware:**
- NVIDIA RTX 5070 (12GB GDDR7) — primary inference and fine-tuning workload
- Intel Core Ultra 5 225F (13 TOPS NPU) — low-power background inference
- 32GB DDR5 RAM (16GB available for system, 16GB for caching)

**Target:** Dual-setup architecture where the RTX 5070 handles heavy inference and fine-tuning, while the Intel NPU provides always-on, low-power background inference for specialized tasks.

## Dual-Device Strategy

```
RTX 5070 (12GB GDDR7) — Primary workload
├── Main inference: Qwen 3 8B Q4_K_M (~50-100 tok/s)
├── Fine-tuning: QLoRA via Unsloth
└── Code model: Qwen 2.5 7B Coder

Intel NPU (13 TOPS) — Background, low power (~5W)
├── Lightweight model via OpenVINO (15-20 tok/s)
└── Use cases: always-on assistant, code completion
```

The 225F has **no integrated GPU**, so the 16GB of available system RAM cannot be used as GPU VRAM. The NPU operates independently using system RAM through OpenVINO or IPEX-LLM, not as an extension of GPU memory.

## Project Structure

```
code-ai-workspace/
├── docs/                  # Documentation (this folder)
├── models/                # Downloaded models
├── datasets/              # Created datasets
│   ├── raw/              # Original books/texts
│   ├── processed/        # Processed data
│   └── training/         # Final training format
├── fine-tuning/          # Training scripts and configs
│   ├── configs/          # Training configuration files
│   └── outputs/          # Checkpoints and trained LoRAs
├── inference/            # Model serving configs
└── tools/                # Utility scripts
```

## Phases (Revised Roadmap)

### Phase 1: Inference Setup
1. Install Ollama (native Windows)
2. Pull models:
   - Qwen 3 8B
   - DeepSeek-R1-Distill-Qwen-7B
   - Qwen 2.5 7B Coder
3. Benchmark performance and latency on RTX 5070
4. Configure network access (API server on port 11434)
5. (Optional) Setup OpenVINO for Intel NPU background inference

**Deliverable:** Running local API accessible from network at `http://<machine-ip>:11434`

### Phase 2: Data Preparation
1. Organize source books/texts into `datasets/raw/`
2. Build text extraction pipeline
3. Synthesize Q&A dataset using LLM (HuggingFace Synthetic Data Generator or Claude API distillation)
4. Format for training (chat/instruction format — ChatML or Alpaca)
5. Split into train/validation sets

**Deliverable:** 5-10K example training dataset in `datasets/training/`

### Phase 3: Fine-Tuning
1. Install WSL2 + CUDA toolkit in WSL2
2. Install Unsloth in WSL2 environment: `pip install unsloth`
3. Configure QLoRA training (see fine-tuning-guide.md for parameters)
4. Run training, monitor VRAM/loss with logging
5. Evaluate resulting model on holdout test set

**Deliverable:** LoRA weights in `fine-tuning/outputs/`

### Phase 4: Deploy
1. Merge LoRA with base model (or keep separate for modularity)
2. Quantize fine-tuned model if needed
3. Serve final version via Ollama or SGLang
4. Setup network API endpoint with authentication if needed

**Deliverable:** Production-ready model serving via API

## Key Constraints

- **12GB VRAM is the hard ceiling** — No memory expansion tricks (225F has no iGPU, so no shared memory override)
- **QLoRA fine-tuning of 7B models uses ~11-12GB VRAM** — Tight, but works with careful configuration
- **Fine-tuning adjusts style/format, not factual knowledge** — Use RAG (Retrieval Augmented Generation) combined with fine-tuning for domain-specific facts
- **WSL2 required for fine-tuning** — Native Windows inference works fine, but training requires Linux environment with proper CUDA setup
- **Sequence length trade-off** — Longer contexts increase quality but consume more VRAM; start at 512 tokens and increase gradually
