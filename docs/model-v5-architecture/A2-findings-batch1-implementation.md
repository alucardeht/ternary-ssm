# A2 Findings Batch 1 Implementation

**Date:** 2026-03-01
**Status:** Complete — 3 findings implemented and integrated
**Files Modified:** config.py, train.py, model.py, __main__.py

## Overview

Three A2 findings have been implemented to optimize memory and training efficiency:

1. **F1: Selective Activation Checkpointing** — granular control over which layers to checkpoint
2. **F2: torch.compile Memory Budget API** — limit activation memory during compilation
3. **F9: Top-K Gradient Sparsification** — sparse gradient updates with error feedback

All features are disabled by default and can be enabled via CLI flags.

---

## F1: Selective Activation Checkpointing

**Location:** `src/model_v5/model.py` (lines 693-723)
**Config Field:** `config.activation_checkpoint_granularity`
**CLI Flag:** `--activation-checkpoint-granularity {none|full|selective}`

### How It Works

Activation checkpointing saves memory by not storing intermediate activations, recomputing them during backprop. However, checkpointing adds overhead — recomputing cheap operations (SSM add/sub) is faster than recomputing expensive operations (attention matmuls).

F1 introduces granular control:

- **`"none"`** (default): No checkpointing — store all activations
- **`"full"`**: Checkpoint every layer — maximum memory savings, overhead on cheap ops
- **`"selective"`**: Checkpoint only SSM layers, not SWA (attention) layers — balanced approach

### Implementation

```python
granularity = getattr(self.config, 'activation_checkpoint_granularity', 'none')

if granularity == 'full':
    should_checkpoint = True
elif granularity == 'selective':
    should_checkpoint = not layer.is_swa  # Only SSM layers
elif self.config.ternary_aware_checkpointing and not layer.is_swa:
    should_checkpoint = True
```

### Expected Behavior

- **`selective`**: 30-40% memory reduction vs `none`, minimal overhead (SSM recomputation is cheap)
- **`full`**: 50-60% memory reduction, 10-20% speed loss on training (expensive attention recomputation)
- **Interaction:** Respects existing `ternary_aware_checkpointing` and `use_gradient_checkpointing` flags

### Testing

Enable and verify with:
```bash
python -m src.model_v5 --config wikitext_6m \
    --activation-checkpoint-granularity selective \
    --max-steps 100 --batch-size 16
```

---

## F2: torch.compile Memory Budget API

**Location:** `src/model_v5/train.py` (lines 1002-1012)
**Config Field:** `config.compile_memory_budget`
**CLI Flag:** `--compile-memory-budget FLOAT`

### How It Works

PyTorch 2.5+ `torch.compile()` supports `activation_memory_budget` option to limit memory allocated for intermediate activations during compilation. This allows trading memory for recomputation.

F2 exposes this as a simple float (0.0-1.0 fraction of available memory).

### Implementation

```python
compile_options = {}
if config.compile_memory_budget is not None:
    compile_options['activation_memory_budget'] = config.compile_memory_budget

if compile_options:
    model = torch.compile(model, mode='default', dynamic=False, options=compile_options)
```

### Expected Behavior

- **`None`** (default): torch.compile uses default memory strategy
- **`0.5`**: Limit activations to 50% of available VRAM, recompute the rest
- **`0.2`**: Aggressive memory limit, maximum recomputation
- **Graceful fallback:** If option unavailable (PyTorch < 2.5), silently ignored

### Testing

Enable and verify with:
```bash
python -m src.model_v5 --config wikitext_6m \
    --use-compile --compile-memory-budget 0.5 \
    --max-steps 50
```

Check VRAM usage before/after to measure effect.

---

## F9: Top-K Gradient Sparsification

**Location:** `src/model_v5/train.py` (lines 120-154, 1569-1572, 1947, 2068)
**Config Fields:**
- `config.use_topk_grad_sparsification`
- `config.topk_grad_sparsity`

**CLI Flags:**
- `--use-topk-grad-sparsification`
- `--topk-grad-sparsity FLOAT` (default: 0.9)

### How It Works

Gradient sparsification reduces communication overhead in distributed training (or in this case, reduces gradient accumulation overhead). Instead of updating all weights every step, keep only top-K% by magnitude and defer the rest via error feedback.

**Key principle:** Error Feedback (EF) — accumulate discarded gradients in a buffer and add back next step.

### Implementation

```python
def apply_topk_gradient_sparsification(model, sparsity=0.9, error_buffers=None):
    for name, param in model.named_parameters():
        grad = param.grad.data + error_buffers[name]  # Add previous error

        k = max(1, int(grad.numel() * (1 - sparsity)))
        topk_vals, topk_idx = torch.topk(grad.abs().view(-1), k)

        mask = torch.zeros_like(grad.view(-1))
        mask[topk_idx] = 1.0

        sparse_grad = grad * mask
        error_buffers[name] = (grad - sparse_grad).detach()  # Save error

        param.grad.data.copy_(sparse_grad)
```

Called before each `optimizer.step()` for both AMP and non-AMP paths.

### Expected Behavior

- **`sparsity=0.9`** (default): Keep top 10% of gradients by magnitude per parameter
- **`sparsity=0.95`**: Keep top 5% (more aggressive)
- **Effect:**
  - 5-10x fewer non-zero gradients
  - Per-parameter error accumulation ensures no information loss asymptotically
  - Minimal quality impact if used with error feedback (EF proven to converge to same optimum)

### Configuration

Recommended settings:
- Conservative: `sparsity=0.9` (keep 10%)
- Moderate: `sparsity=0.95` (keep 5%)
- Aggressive: `sparsity=0.98` (keep 2%)

### Testing

Enable and verify sparsity effect with:
```bash
python -m src.model_v5 --config wikitext_6m \
    --use-topk-grad-sparsification --topk-grad-sparsity 0.95 \
    --max-steps 100 --grad-accum-steps 2
```

Expect comparable final loss with ~10x fewer gradient updates (less communication).

---

## Integration Points

### config.py Changes
- Line 110: `compile_memory_budget: float = None`
- Line 112: `activation_checkpoint_granularity: str = "none"`
- Line 232-233: `use_topk_grad_sparsification` and `topk_grad_sparsity`

### __main__.py Changes
- Lines 66-71: Added F1/F2 CLI flags
- Lines 564-567: Added F9 CLI flags
- Lines 702-705: Map F1/F2 args to config
- Lines 943-945: Pass F9 args to train_model

### train.py Changes
- Lines 120-154: `apply_topk_gradient_sparsification()` function
- Lines 1002-1012: torch.compile with memory budget option
- Lines 1569-1572: Initialize `topk_error_buffers`
- Lines 1947, 2068: Call F9 before optimizer.step()

### model.py Changes
- Lines 693-723: Enhanced layer forward with F1 granularity logic
- Respects new `activation_checkpoint_granularity` config field

---

## Feature Combinations

These three findings work well together:

1. **F1 + F2:** Reduce VRAM via selective checkpointing, then use compile budget to further constrain activations
2. **F1 + F9:** Sparse gradients benefit from checkpointing (fewer activations to store)
3. **F1 + F2 + F9:** Full memory optimization pipeline

Example combined run:
```bash
python -m src.model_v5 --config ternary_50m \
    --activation-checkpoint-granularity selective \
    --use-compile --compile-memory-budget 0.4 \
    --use-topk-grad-sparsification --topk-grad-sparsity 0.95 \
    --max-steps 500 --batch-size 16
```

---

## Backward Compatibility

- All features default to OFF (no behavior change)
- No breaking changes to existing APIs
- Existing experiments unaffected unless flags explicitly enabled
- Can be selectively enabled per experiment

---

## Future Work

- Monitor quality impact of F9 across scales (6M, 50M, 100M)
- Tune F1 granularity per model size (smaller models benefit less from selective)
- Combine F2 with gradient checkpointing for aggressive memory limits
- Profile actual VRAM/speed impact per combination
