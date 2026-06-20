# Execution Guide — Agent Reference

**Read this before executing any command or modifying code in this project.**

## Environment

- **OS:** Windows 11 Pro (bash shell via Git Bash)
- **Python:** 3.12.10 at `C:\Users\th1rt\AppData\Local\Programs\Python\Python312\python.exe`
- **IMPORTANT:** Use `python` command directly. Do NOT install Python. It is already installed.
- **PyTorch:** 2.10.0+cu128
- **GPU:** NVIDIA GeForce RTX 5070 (12GB GDDR7, CUDA available)
- **HuggingFace datasets:** 4.5.0
- **Working directory:** `C:\Users\th1rt\Documents\Workspaces\code-ai-workspace`

## How to Run Experiments

All experiments are Python modules. Run from the workspace root:

```bash
cd /c/Users/th1rt/Documents/Workspaces/code-ai-workspace
python -m src.model_v2.experiments.<experiment_name> <args>
```

Examples:
```bash
# D.1 Scalar Learning
python -m src.model_v2.experiments.bench_phase10_d1_scalar_learning baseline steps=5000

# B.1 Direct Ternary
python -m src.model_v2.experiments.bench_phase10_b1_direct_ternary progressive steps=5000
```

**CRITICAL RULES:**
- NEVER use `run_in_background: true` for training/benchmarks
- NEVER set timeout > 120000ms
- Each experiment segment must complete in <2 minutes
- If a segment takes >2 min, reduce steps (use 200-500 steps per segment)
- After each run, check GPU with `nvidia-smi` before starting next
- Set `device='cuda'` explicitly when needed
- Max VRAM: ~10.8GB. Max RAM: ~16GB.

## Project Structure

```
src/model_v2/
├── config.py              # Model configurations (6M, 50M, 100M scales)
├── data.py                # Data loading (WikiText-103)
├── model.py               # Main model architecture
├── mup.py                 # μP (maximal update parameterization)
├── train.py               # Training utilities
├── prepare_wikitext.py    # Data preparation
├── components/
│   ├── attention.py       # Sliding window attention
│   ├── rmsnorm.py         # RMSNorm
│   ├── router.py          # MoD routing
│   ├── rwkv7.py           # RWKV-7 implementation
│   ├── ssm.py             # SSM/SSD (Mamba-2 style)
│   └── ternary.py         # TernaryLinear (STE, HESTIA, alpha)
├── experiments/
│   ├── bench_phase10_*.py # Phase 10 Discovery experiments
│   └── ...                # Earlier phase experiments
└── kernels/               # Custom CUDA kernels
```

## Key Components

### TernaryLinear (`components/ternary.py`)
- Weights quantized to {-1, 0, +1} via STE
- Has `quantize_enabled` flag (True/False)
- Has optional `alpha` parameter (per-layer scalar)
- Has optional `freeze_patterns` mode
- HESTIA soft quantization with temperature annealing

### Config scales
- `config_wikitext_bench()` — 6M params (quick tests)
- `config_wikitext_50m()` — 50M params (scale validation)

## Checkpoints

Saved to `.checkpoints/` directory at workspace root.
Format: `<experiment>_<variant>_<steps>.pt`
Contains: model state, optimizer state, scaler, scheduler, step, losses list.

## Dataset

WikiText-103 (auto-downloads via HuggingFace datasets on first use).
Cached at `~/.cache/huggingface/`.
