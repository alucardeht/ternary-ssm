# RTKI — Runtime Knowledge Integration System (TIER 2-D)

## Overview

RTKI enhances GTF (Greedy Ternary Flipping) with sparse residual tracking to recover quantization information loss. The system combines gradient-based signals with residual magnitude prioritization for more effective ternary weight adaptation.

**Key insight:** Ternary quantization (FP32 → {-1,0,+1}) discards information. Sparse residuals capture this loss and guide smarter weight flips.

## Mathematical Foundation

### The Problem: Quantization Information Loss

When ternary quantizing FP32 shadow weights:
- Original weight: `w_float ∈ ℝ`
- Quantized weight: `w_ternary ∈ {-1, 0, +1}`
- Information loss: `r = w_float - w_ternary` (residual)

The residual `r` represents the quantization error — how much information was discarded during quantization.

### The Solution: Sparse Residual Tracking

Instead of tracking all residuals (expensive), maintain **sparse representation** of top-k largest residuals:
- Per module: Store indices and magnitudes of top-1% residuals (configurable)
- Quality metric: Mean absolute residual per module indicates quantization pressure
- Flip prioritization: Combine gradient signal + residual magnitude

### Flip Scoring Formula

Traditional GTF scores flips using gradient signal alone:
```
score_gtf = ∇L · Δw
```

RTKI enhances this with residual information:
```
residual_score = |w_float - w_ternary| / max(|residuals|)  [normalized to [0,1]]

combined_score = (1 - α) · score_gtf + α · residual_score
```

Where `α = residual_weight` (default 0.3 = 70% gradient, 30% residual).

**Interpretation:** High residual magnitude = position where quantization hurt quality most = should prioritize flipping.

## System Architecture

### SparseResidualTracker

Tracks quantization residuals for all ternary modules.

**Workflow:**
1. Iterate over all ternary modules (detected via `hasattr(module, 'quantize_enabled')`)
2. For each module:
   - Compute quantized weights using threshold: `quantized = sign(w) * (|w| > threshold)`
   - Compute residual: `residual = shadow_weight - quantized`
   - Compute quality metric: `quality = mean(|residual|)`
   - Select top-k residuals (sparse representation)
3. Store sparse residuals (indices, values) for later use

**Key methods:**
- `update()` — Recompute residuals from current shadow weights
- `get_residuals(module_name)` — Return (indices, values) for sparse residuals
- `get_quality_map()` — Return dict of mean-abs-residual per module
- `apply_corrections(alpha)` — Context manager for inference-time corrections

### RTKIAdapter

Combines GTF flip evaluation with residual prioritization.

**Workflow:**
1. Update residual tracker (recompute residuals)
2. Run GTF evaluate_flips() to get gradient-based scores
3. Compute normalized residual scores per weight
4. Combine scores: `combined = (1-α) * grad_score + α * res_score`
5. Sort flips by combined score
6. Apply top-k flips within budget

**Key methods:**
- `evaluate_and_flip(data_loader, budget, n_batches, device)` — Evaluate and apply flips
- `update_residuals()` — Update tracker after training
- `get_stats()` — Return combined statistics

### _ResidualCorrectionContext

Context manager for inference-time residual corrections.

Allows temporary addition of sparse residuals during inference to recover some quantization quality:
```python
with tracker.apply_corrections(alpha=0.1):
    output = model(input_ids)  # With corrections applied
```

## Implementation Details

### File Structure

```
src/model_v5/adapters/
├── gtf.py          # Existing GTF adapter
├── rtki.py         # RTKI system (NEW)
└── __init__.py     # Exports: GTFAdapter, RTKIAdapter, SparseResidualTracker
```

### Key Design Decisions

1. **Sparse representation, not dense:** Only track top-1% residuals per module to minimize memory overhead
2. **Threshold-based quantization detection:** Leverage module's existing `threshold_coeff` attribute
3. **Composition over inheritance:** RTKI wraps GTFAdapter internally, adding residual layer
4. **No model modification:** All operations read-only until flips are applied (same as GTF)

### Memory Efficiency

For a module with 100K weights and top_k_fraction=0.01:
- Sparse residual storage: 1K indices + 1K floats ≈ 8KB per module
- vs. Dense residuals: 100K floats ≈ 400KB per module
- **Savings: 50x less memory**

## Usage Examples

### Basic Evaluation and Flipping

```python
from src.model_v5.adapters import RTKIAdapter

model = create_model()
optimizer = TernarySignOptimizer(model.parameters(), lr=0.001)
data_loader = create_data_loader()

rtki = RTKIAdapter(model, optimizer, residual_weight=0.3)

stats = rtki.evaluate_and_flip(
    data_loader,
    budget=0.001,      # Flip 0.1% of ternary params
    n_batches=5,       # Use 5 batches for evaluation
    device='cuda'
)

print(f"Applied {stats['applied']} flips")
print(f"Quality map: {rtki.get_stats()['quality_map']}")
```

### Periodic Updates During Training

```python
for step in range(total_steps):
    loss = train_step()

    if step % 1000 == 0 and step > 0:
        rtki.update_residuals()
        quality = rtki.tracker.get_quality_map()
        print(f"Quantization quality at step {step}: {quality}")
```

### Inference-Time Corrections

```python
with rtki.tracker.apply_corrections(alpha=0.1):
    logits = model(input_ids)  # Quality recovery: residuals added
    probs = softmax(logits)
```

## Hyperparameter Tuning

### top_k_fraction (default 0.01)
- **Higher:** More residuals tracked, better accuracy but more memory
- **Lower:** Fewer residuals, faster, but may miss important positions
- **Recommended range:** 0.005 to 0.05

### residual_weight (default 0.3)
- **Higher (0.5-0.7):** More emphasis on residual magnitude, less on gradients
- **Lower (0.1-0.2):** More emphasis on gradient signal
- **Recommended:** Start at 0.3, adjust based on quality improvements

### budget (default 0.001)
- **Higher:** Apply more flips, faster adaptation but higher forgetting risk
- **Lower:** More conservative, slower adaptation but safer
- **Recommended:** 0.001-0.005 per adaptation cycle

## Validation & Testing

Comprehensive test suite at `src/model_v5/experiments/test_tier2d.py`:

1. **SparseResidualTracker test:**
   - Verify residuals computed correctly (non-zero for quantized weights)
   - Verify quality_map has entries for all ternary modules
   - Verify sparse representation working (top-k selection)

2. **RTKIAdapter test:**
   - Verify flip evaluation produces scores
   - Verify flips applied correctly
   - Verify combined stats contain both gradient and residual info

3. **Context manager test:**
   - Verify apply_corrections context manager works
   - Verify correction tensors match module shapes

**Run tests:**
```bash
python -m src.model_v5.experiments.test_tier2d
```

## Architectural Alignment

### Integration with TIER 2 Framework

RTKI is component 4 of TIER 2 (domain adaptation):
- **TIER 2-A:** Progressive depth growing ✓
- **TIER 2-B:** Data efficiency (curriculum + core-set)
- **TIER 2-C:** Distributed (Local SGD + compression)
- **TIER 2-D:** RTKI (GTF enhancement + residuals) ← **This**

### Compatibility with Existing Systems

- **Works with GTF:** RTKI wraps GTFAdapter, fully compatible
- **Works with TernarySSM:** Uses standard `quantize_enabled` + `group_size` detection
- **Works with any optimizer:** Supports TernarySignOptimizer and others
- **Zero model modifications:** Read-only until flips applied (safe)

## Future Extensions

1. **Adaptive alpha:** Learn residual_weight during adaptation instead of fixed
2. **Hierarchical residuals:** Track residuals at multiple scales
3. **Gradient matching:** Use residuals to match gradient statistics across ternary + FP32
4. **Inference recovery:** Learned sparse residual networks for better quality recovery

## References

- **GTF (Greedy Ternary Flipping):** `src/model_v5/adapters/gtf.py`
- **Ternary Quantization:** `src/model_v5/components/ternary.py`
- **TIER 2 Design:** `docs/model-v5-architecture/vision-and-status.md`
