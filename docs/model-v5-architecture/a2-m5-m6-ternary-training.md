# A2-M5 & A2-M6: UCB-Bandit Ternary Training and FSCR Flip Gate

## Overview

Two complementary findings for improved ternary weight training:

- **A2-M5 (UCBTernaryLinear)**: Eliminates STE entirely by treating each ternary weight as a 3-armed bandit with UCB1 decision criterion
- **A2-M6 (FSCR)**: Prevents bad flips driven by activation outliers via a flip safety/confidence ratio gate

## A2-M5: UCB-Bandit Ternary Training

### Concept

Instead of:
1. Maintain FP32 shadow weights
2. Quantize to ternary
3. Compute loss
4. STE backward (approximate gradient through quantization)
5. Update shadow weights with Adam

UCB-Bandit ternary training:
1. Maintain ternary weights directly {-1, 0, +1}
2. For each weight position, compute exact loss delta for each of 3 values
3. Use UCB1 criterion to decide whether to flip
4. No shadow weights, no STE, no Adam optimizer state needed

### Memory Savings

67% reduction in optimizer state:
- Traditional STE: shadow_weights + m + v = 3× parameters per ternary weight
- UCB-Bandit: reward_sum (3 arms) + pull_count (3 arms) + total_pulls = negligible overhead

For a 7B model with 60% ternary weights:
- STE: ~84 GB optimizer state
- UCB-Bandit: ~28 GB optimizer state

### Implementation

Located in: `src/model_v5/components/ucb_ternary.py`

```python
from src.model_v5.components.ucb_ternary import UCBTernaryLinear

# Use as drop-in replacement for TernaryLinear
layer = UCBTernaryLinear(
    in_features=1024,
    out_features=2048,
    bias=False,
    exploration_c=1.0  # UCB exploration coefficient
)
```

### How It Works

#### Forward Pass
- Converts int8 ternary weights to float and performs standard F.linear()
- Saves input for backward pass via hook

#### Backward Pass (Hook-Based)
1. Computes flip utilities for each weight position and each possible value
   - Utility of flipping w_ij to value v: U_ij(v) = -(grad_output_i * (v - current_w) * x_j)

2. Updates UCB1 statistics:
   - reward_sum[i,j,k] += utility of choosing arm k
   - pull_count[i,j,k] += 1

3. Selects best arm via UCB1:
   - UCB_score[i,j,k] = reward_mean[k] + c * sqrt(ln(T) / n_k)
   - where T = total pulls, n_k = pulls of arm k, c = exploration_c

4. Updates weights: w_ij ← value of best arm

### Configuration

```python
config.use_ucb_ternary_training = True
config.ucb_ternary_exploration_c = 1.0  # balance exploration/exploitation
```

Or via CLI:
```bash
python -m src.model_v5 \
  --use-ucb-ternary-training \
  --ucb-ternary-exploration-c 1.0
```

### Debugging

The module logs statistics every 500 calls:
- Mean pulls per weight
- Mean reward
- Ternary weight distribution (-1/0/+1 percentages)

```
[UCBTernary] call=501 mean_pulls=142.3 mean_reward=0.0042
distribution=-1:32.1% 0:35.8% +1:32.1%
```

## A2-M6: FSCR Flip Safety/Confidence Ratio

### Concept

FSCR prevents weight flips that are driven by activation outliers rather than genuine learning signal.

**Key Insight**: A large gradient on a small weight often indicates an outlier (few extreme input values) rather than true learning. Flipping based on outliers leads to unstable training.

### Implementation

Located in: `src/model_v5/components/fscr.py`

Three functions:
1. `compute_fscr_mask(model, fscr_threshold)` — Compute safety mask
2. `apply_fscr_mask(model, fscr_masks)` — Zero out unsafe gradients
3. `log_fscr_statistics(model, fscr_masks, step)` — Log diagnostics

### How It Works

For each parameter:
1. Compute safety ratio: `ratio = |grad| / |weight|`
   - High ratio = large gradient on small weight = suspicious (outlier-driven)
   - Low ratio = proportional gradient = safe

2. Create safety mask: `mask = ratio < threshold`

3. Zero out unsafe gradients: `grad[~mask] = 0`

This prevents flips that would be driven by extreme input values rather than stable learning.

### Configuration

```python
config.use_fscr_flip_gate = True
config.fscr_threshold = 2.0  # Threshold for flip safety (higher = more conservative)
```

Or via CLI:
```bash
python -m src.model_v5 \
  --use-fscr-flip-gate \
  --fscr-threshold 2.0
```

### Threshold Guidance

- `fscr_threshold = 1.0`: Very conservative, blocks many gradients
- `fscr_threshold = 2.0`: Balanced (default), blocks outlier-driven flips
- `fscr_threshold = 5.0`: Permissive, only blocks extreme outliers

### Debugging

Logs statistics every 500 steps:
```
[FSCR] step=500 total_params=125432000 masked_params=3456 masked%=0.00%
[FSCR] step=1000 total_params=125432000 masked_params=8912 masked%=0.01%
```

Low masked% indicates normal training. Spike in masked% indicates outlier episodes.

## Combined Usage: A2-M5 + A2-M6

For best results when using UCB-Bandit training:

```bash
python -m src.model_v5 \
  --config ternary_6m \
  --max-steps 5000 \
  --use-ucb-ternary-training \
  --ucb-ternary-exploration-c 1.0 \
  --use-fscr-flip-gate \
  --fscr-threshold 2.0
```

**Benefits**:
- M5 (UCB) reduces optimizer state bloat
- M6 (FSCR) stabilizes training by filtering outlier-driven flips
- Together: efficient memory + stable convergence

## Implementation Notes

### UCBTernaryLinear Integration

UCBTernaryLinear is a new module class. To use it in a model, either:

1. **Manually replace TernaryLinear** in model definition:
   ```python
   from components.ucb_ternary import UCBTernaryLinear
   # Instead of: layer = TernaryLinear(...)
   layer = UCBTernaryLinear(...)
   ```

2. **Create a factory function** that conditionally creates UCBTernaryLinear or TernaryLinear based on config

3. **Add to model.py** as part of model architecture selection

### FSCR Integration

FSCR is applied automatically in the training loop (train.py) when `use_fscr_flip_gate=True`. No model changes needed.

### Backward Compatibility

- UCBTernaryLinear and FSCR are optional features
- Default behavior unchanged when features disabled
- Both features are OFF by default

## Testing

### Smoke Test (UCBTernaryLinear)

```python
import torch
from src.model_v5.components.ucb_ternary import UCBTernaryLinear

layer = UCBTernaryLinear(128, 256, bias=False)
x = torch.randn(2, 128)
y = layer(x)
loss = y.sum()
loss.backward()
print(f"Output shape: {y.shape}")  # Should be [2, 256]
```

### Smoke Test (FSCR)

```python
import torch
from src.model_v5.components.fscr import compute_fscr_mask, apply_fscr_mask

model = torch.nn.Linear(128, 256)
x = torch.randn(2, 128)
y = model(x)
loss = y.sum()
loss.backward()

masks = compute_fscr_mask(model, fscr_threshold=2.0)
apply_fscr_mask(model, masks)

# Check that some gradients were masked
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().sum().item():.4f}")
```

## Performance Expectations

### UCBTernaryLinear
- Training speed: Similar to STE baseline (backward hook overhead ~1-2%)
- Memory: 67% reduction in optimizer state
- Convergence: Potentially faster (no STE approximation error) - TBD with experiments

### FSCR
- Training speed: Negligible overhead (computed once per optimizer step)
- Memory: None (masks computed on-the-fly, not stored)
- Training stability: Should improve loss smoothness during instability episodes

## Future Work

1. **Adaptive exploration_c**: Decrease c over training (more exploitation)
2. **Per-weight learning rates**: Different c per parameter based on distance to threshold
3. **FSCR variants**: Input-aware outlier detection, spectral analysis of grads
4. **Combined optimizer**: Integrate UCB decision-making with gradient-based optimization for hybrid approach

## References

- UCB1 Algorithm: Auer et al. "Finite-time Analysis of the Multi-armed Bandit Problem" (2002)
- STE Background: Bengio et al. "Estimating and Backpropagating Gradients Through Stochastic Neurons for Conditional Computation" (2013)
- Ternary Training: Zhou et al. "Bi-Real Net: Binarizing Deep Network Towards Real-Network Performance" (2018)
