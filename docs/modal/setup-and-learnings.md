# Modal.com — Setup, Configuration & Learnings

## Account & Auth

- **Profile:** juliolena25
- **Plan:** Starter (free, $30/month credits)
- **CLI:** `pip install modal` (v1.3.3)
- **Auth:** `modal token set --token-id <id> --token-secret <secret>`
- **Config file:** ~/.modal.toml

## Volumes Created

| Volume | Purpose | Size |
|--------|---------|------|
| ternary-ssm-data | WikiText-103 tokenized (.bin files) | ~221 MB |
| ternary-ssm-checkpoints | Training checkpoints and results | variable |

## GPU Pricing (February 2026)

| GPU | $/hour | VRAM | Notes |
|-----|--------|------|-------|
| T4 | $0.59 | 16 GB | No native BF16 |
| L4 | $0.80 | 24 GB | Ada Lovelace, BF16 OK |
| A10 | $1.10 | 24 GB | |
| A100 40GB | $2.10 | 40 GB | |
| A100 80GB | $2.50 | 80 GB | |
| L40S | $1.95 | 48 GB | |
| H100 | $3.95 | 80 GB | SXM variant |
| H200 | $4.54 | 141 GB | |
| B200 | $6.25 | 192 GB | Blackwell |

- Billing by the second (no minimums)
- Non-preemptible = 3x price
- Region selection = 1.25-2.5x price

## Key Commands

```bash
# Run (blocking, shows logs)
modal run script.py::function_name --arg value

# Run detached (safe to close terminal)
modal run --detach script.py::function_name --arg value

# Check logs
modal logs <app-id>

# Volume operations
modal volume ls <volume-name> /path/
modal volume put <volume-name> local/path/ /remote/path/
modal volume get <volume-name> /remote/path/ local/path/
```

## Data Upload

```bash
# WikiText-103 (one-time, ~221 MB)
modal volume create ternary-ssm-data
modal volume put ternary-ssm-data data/wikitext103/ /wikitext103/
```

Files on volume:
- /wikitext103/train.bin (217.6 MB)
- /wikitext103/val.bin (467 KB)
- /wikitext103/test.bin (532 KB)
- /wikitext103/tokenizer.json (2.2 MB)

## Training Script

Main file: `modal_train.py` (project root)

Usage:
```bash
# FP16 baseline (1 epoch, ~50K steps)
modal run --detach modal_train.py::main --mode fp16 --epochs 1

# Ternary (5 epochs)
modal run --detach modal_train.py::main --mode ternary --epochs 5

# Download results
modal volume get ternary-ssm-checkpoints /checkpoints/ ./modal-results/
```

## Learnings & Rules

### RULE 1: Always test locally before sending to Modal
The script MUST be tested locally (at least model creation + 1 training step) before running on Modal. Debugging on Modal wastes money and is slow.

Local test command:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from modal_train import *
# Test model creation, data loading, one step
# WITHOUT actually calling Modal
"
```

### RULE 2: Image build is cached
First run builds the Docker image (downloads PyTorch ~2GB). Subsequent runs reuse the cached image. If you change pip dependencies, the image rebuilds.

### RULE 3: Volume commits
- Background auto-commit every few seconds
- Call `volume.commit()` explicitly after important checkpoints
- Container B must call `volume.reload()` to see container A's writes

### RULE 4: Code mount vs image install
- `modal.Mount.from_local_dir()` mounts code at runtime (no rebuild needed)
- Changes to src/ are picked up immediately on next run
- Changes to pip dependencies require image rebuild

### RULE 5: Preemption handling
- Modal can preempt containers (cheaper rate)
- Use `retries=modal.Retries(max_retries=10)` for automatic restart
- Checkpoint-based resume is essential for long training

### RULE 6: torch.compile warmup
- First batch with torch.compile takes 5-10 minutes
- This is compilation overhead, not a bug
- Subsequent batches run at full speed

## Cost Tracking

| Date | Test | GPU | Duration | Cost Est. | Notes |
|------|------|-----|----------|-----------|-------|
| 2026-02-19 | Debug runs (10x) | L4 | <1 min each | ~$0.15 total | Errors 1-6 |
| 2026-02-19 | FP16 1ep 50M b32 | L4 | ~10 min | ~$0.13 | Val PPL 55.22, Test PPL 57.68 |
| 2026-02-19 | Ternary 5ep 50M b32 | L4 | ~51 min | ~$0.68 | Val PPL 34.33, Test PPL 36.23 |
| 2026-02-19 | FP16 5ep 50M b32 | L4 | ~51 min | ~$0.68 | Val PPL 31.96, Test PPL 33.97 |
| | | | **Total** | **~$1.64** | All 3 tests completed successfully |

## API Reference (Modal SDK 1.0+ — Verified Feb 2026)

### CRITICAL: Modal SDK 1.0 (May 2025) introduced breaking changes

Many online examples use outdated API. Below are the CORRECT patterns for SDK 1.0+.

### Image & Source Code Mounting

**CORRECT pattern:**
```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu124", "numpy", extra_index_url="https://download.pytorch.org/whl/cu124")
    .add_local_dir("src", remote_path="/code/src")
)
```

**Available Image methods for local files:**
- `Image.add_local_dir(local_path, remote_path=...)` — PRIMARY way to include directories
- `Image.add_local_file(local_path, remote_path=...)` — single file
- `Image.add_local_python_source("package")` — Python packages

**DEPRECATED (removed in 1.0):**
- `modal.Mount` — no longer exists
- `mount=` / `mounts=` parameters — removed
- `Image.copy_mount()` — removed
- `context_mount=` — removed

### App Definition

```python
app = modal.App("name", include_source=True)
```

**`include_source` parameter:**
- `True` — includes the defining module (the .py file with @app.function decorators). Use this when the defining module is NOT explicitly added to the image.
- `False` — ONLY use when you have explicitly added the defining module to the image via `Image.add_local_file()` or similar.
- `None` (default if omitted) — legacy automount behavior, will break in future 1.x versions

**Why `True` is the default:**
The defining module file (modal_train.py with @app.function) must be in the container for Modal to find the decorated functions. `Image.add_local_dir()` adds project packages (src/) but NOT the defining module itself. They serve different purposes:
- `include_source=True` → includes modal_train.py in the container
- `Image.add_local_dir("src", remote_path="/code/src")` → includes src/ package for imports

Both are needed and are NOT redundant.

### GPU Types (String Syntax — Object Syntax Deprecated)

```python
@app.function(gpu="L4")  # CORRECT
@app.function(gpu=modal.gpu.L4())  # WRONG (deprecated)
```

**Available GPU types (string format):**
- `"T4"`, `"L4"`, `"A10G"`, `"A100"`, `"A100-40GB"`, `"A100-80GB"`
- `"L40S"`, `"H100"`, `"H200"`, `"B200"`

**Multi-GPU syntax:**
```python
@app.function(gpu="H100:8")       # 8x H100
@app.function(gpu="A100-80GB:2")  # 2x A100-80GB
```

**Priority scheduling:**
```python
@app.function(gpu="H100!")  # exclamation mark for priority queue
```

**Fallback list:**
```python
@app.function(gpu=["H100", "A100-80GB", "any"])  # try H100, fallback to A100-80GB, then any available
```

### Volumes

```python
vol = modal.Volume.from_name("name", create_if_missing=True)
```

**Important patterns:**
- `.commit()` — MANDATORY after writes, or changes are LOST
- `.reload()` — refresh in another container to see committed changes
- `create_if_missing=True` — safe for first run

**DEPRECATED:**
- `.lookup()` — replaced by `.from_name()`

### Function Decorator

```python
@app.function(
    gpu="L4",
    timeout=3600,
    startup_timeout=120,
    image=image,
    volumes={"/data": data_volume},
)
def my_function():
    pass
```

**New parameter (v1.0+):**
- `startup_timeout` — separate timeout for container initialization (helpful for slow image builds)

### sys.path for imports inside Modal

```python
@app.function(...)
def my_func():
    import sys
    sys.path.insert(0, "/code")  # matches remote_path parent from add_local_dir()
    from src.model_v2.config import config_wikitext_50m
```

### Breaking Changes Summary (May 2025 → SDK 1.0)

| Feature | Old API | Status | Fix |
|---------|---------|--------|-----|
| Code mounting | `modal.Mount.from_local_dir()` | Removed | Use `Image.add_local_dir()` |
| Function mounts | `mount=`, `mounts=` params | Removed | Use `volumes=` instead |
| Image copying | `Image.copy_mount()` | Deprecated | Use `Image.add_local_dir()` |
| Automounting | `MODAL_AUTOMOUNT` env var | Removed | Explicit `include_source=False` |
| GPU selection | `modal.gpu.L4()` object syntax | Deprecated | Use string `"L4"` instead |
| Volume lookup | `.lookup()` method | Replaced | Use `.from_name()` instead |

### Available top-level modal attributes (v1.0+)

- `modal.App` — application definition
- `modal.Image` — container image builder
- `modal.Volume` — persistent network storage
- `modal.CloudBucketMount` — S3/GCS mount
- `modal.Secret` — secrets management

## Errors & Fixes

### Error 1: ModuleNotFoundError: No module named 'src'
**Date:** 2026-02-19
**Cause:** Missing `src/__init__.py` + wrong code mount approach
**Fix:** Created `src/__init__.py` + used `Image.add_local_dir()` instead of `modal.Mount`
**Cost wasted:** ~$0.01 (image build only, training never started)

### Error 2: AttributeError: module 'modal' has no attribute 'Mount'
**Date:** 2026-02-19
**Cause:** `modal.Mount` was deprecated/removed in Modal v1.3.3. Used outdated API from old docs/examples.
**Fix:** Use `Image.add_local_dir("src", remote_path="/code/src")` instead
**Lesson:** ALWAYS check actual installed API, not web examples that may be outdated

### Error 3: ModelConfig Wrong Parameters
**Date:** 2026-02-19
**Error:** `TypeError: ModelConfig.__init__() got an unexpected keyword argument 'hidden_dim'`
**Cause:** Code-writer created manual ModelConfig with wrong parameter names (hidden_dim, num_layers, num_heads, num_experts) instead of using the factory function config_wikitext_50m()
**Fix:** Replace manual config with `config = config_wikitext_50m()` and set flags separately
**Lesson:** Always use existing factory functions (config_wikitext_50m, config_wikitext_100m) instead of constructing ModelConfig manually. The factory functions know the correct parameter names and default values.

### Error 4: include_source=False Removes Defining Module
**Date:** 2026-02-19
**Error:** `ModuleNotFoundError: No module named 'modal_train'`
**Cause:** Setting `include_source=False` on `modal.App()` prevents Modal from including the script file (modal_train.py) itself in the container. The container can't import the module to find decorated functions.
**Fix:** Use `include_source=True`. This includes ONLY the defining module (modal_train.py), not other local packages. Other source code (like src/) should be added separately via `Image.add_local_dir()`.
**Lesson:** `include_source=True` and `Image.add_local_dir()` are NOT redundant — they serve different purposes. `include_source=True` = the script with @app.function decorators. `add_local_dir()` = project source packages.

### Error 5: Data Volume Empty After Previous Upload
**Date:** 2026-02-19
**Error:** `FileNotFoundError: /data/wikitext103/train.bin not found`
**Cause:** The data volume `ternary-ssm-data` was empty despite a previous upload. Volumes can be lost if recreated or if the upload wasn't properly committed.
**Fix:** Re-upload local data with `modal volume put ternary-ssm-data data/wikitext103 wikitext103`
**Lesson:** Always verify volume contents with `modal volume ls <name> <path>` before running expensive GPU jobs. Data persistence on Modal volumes should never be assumed — always verify first.

### Error 6: UnboundLocalError — Steps Calculation Before Config
**Date:** 2026-02-19
**Error:** `UnboundLocalError: cannot access local variable 'config' where it is not associated with a value`
**Cause:** Dynamic steps_per_epoch calculation used `config.seq_len` but was placed BEFORE config creation in the function body. Local test script created config separately, masking the ordering issue.
**Fix:** Replace `config.seq_len` with literal `256` in the steps calculation.
**Lesson:** When modifying function internals, verify execution ORDER. Local test scripts may create objects in different order than the actual function. Always re-test the ACTUAL function locally after fixing.

### Error 7: CUDA OOM with batch_size=64 on L4
- **Error**: CUDA Out of Memory with batch_size=64 on L4 (24GB VRAM)
- **Cause**: 72M param model with batch=64, seq_len=256, torch.compile, and BF16 autocast exceeds L4 VRAM. Estimated ~14GB for model+optimizer+activations, but torch.compile overhead pushes it over 24GB.
- **Fix**: Reduced batch_size to 32. Peak VRAM: 11.2-11.4 GB (~47% of L4's 24GB).
- **Lesson**: Always account for torch.compile memory overhead (~2x activations) when estimating VRAM. batch_size=32 is the sweet spot for our 72M model on L4.

### Error 8: FP16 5ep Checkpoint Corrupted on Download
- **Error**: `RuntimeError: PytorchStreamReader failed reading file data/68: invalid header or archive is corrupted`
- **Cause**: The file `modal-results/50m_fp16_5e_b32.pt` (866MB) was corrupted during download from Modal volume
- **Fix**: Re-download with `PYTHONIOENCODING=utf-8 modal volume get ternary-ssm-checkpoints 50m_fp16_5e_b32.pt modal-results/`
- **Status**: Ternary checkpoint (50m_ternary_5e_b32.pt) loads fine. FP16 needs re-download.
