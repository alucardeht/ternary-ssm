# Environment Setup — ML Research Workspace

## Current Working Environment

| Component | Version | Status |
|-----------|---------|--------|
| OS | Windows 11 Pro 10.0.26200 | OK |
| Python | 3.12.10 (venv at `.venv/`) | OK |
| PyTorch | 2.7.0+cu128 | OK |
| torchvision | 0.22.0+cu128 | OK |
| torchaudio | 2.7.0+cu128 | OK |
| numpy | 2.4.2 | OK |
| psutil | 7.2.2 | OK |
| CUDA Toolkit | 12.8 (bundled with PyTorch) | OK |
| NVIDIA Driver | 591.86 (supports CUDA 13.1) | OK |
| GPU | NVIDIA GeForce RTX 5070 (SM_120 Blackwell) | OK |
| GPU VRAM | 12 GB GDDR7 | OK |
| RAM | 32 GB DDR5 | OK |
| CPU | Intel Core Ultra 5 225F (no iGPU) | OK |

## GPU Support History

The RTX 5070 uses Blackwell architecture (SM_120). Initial setup used PyTorch 2.10.0+cu126 which only supported up to SM_90 — GPU was not usable.

**Solution found:** PyTorch 2.7.0 with cu128 (CUDA 12.8) supports SM_120 natively.

Install command:
```
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verified working:** GPU detection, CUDA operations, tensor computation all confirmed on 2026-02-12.

## Virtual Environment

- Location: `C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\.venv\`
- Python: `.venv\Scripts\python.exe`
- Pip: `.venv\Scripts\pip.exe`
- Activation: `scripts\activate_env.bat`

## Key Notes

- PyTorch cu126 does NOT work with RTX 5070 — must use cu128+
- First `import torch` with CUDA can take 10-30 seconds (one-time initialization)
- Training experiments should use `device='cuda'` (not 'cpu')
- Memory guard set at 12 GB (matches GPU VRAM)
