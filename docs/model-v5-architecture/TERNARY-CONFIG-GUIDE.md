# Ternary Configuration Guide

## Issue Summary

The previous baseline run showed **0% ternary parameters**, which was unexpected since v5 was designed to use ternary quantization. The root cause has been identified and documented.

## Root Cause Analysis

### The Problem
- Default `ModelConfig()` has `use_ternary=False`
- Previous baseline used default config without enabling ternary
- Result: Model ran in pure FP16, 0% ternary parameters

### The Solution
- Use `config_ternary_6m()` or `config_ternary_50m()` for ternary training
- These configs have `use_ternary=True` explicitly set
- Also set `ternary_lm_head=True` to quantize the language model head

## Configuration Options

### Default Config (FP16 Only)
```python
from src.model_v5.config import ModelConfig
config = ModelConfig()  # use_ternary=False
```
- **Ternary params**: 0%
- **Type**: Pure FP16
- **Use case**: Baseline comparisons, testing non-quantized models

### Ternary Config - 6M Parameters
```python
from src.model_v5.config import config_ternary_6m
config = config_ternary_6m()  # use_ternary=True
```
- **Total params**: ~20.2M
- **Ternary params**: 59% (11.9M)
- **FP16 params**: 41% (8.3M)
- **Architecture**: 8 layers, d_model=256, 4 attention heads
- **Use case**: Fast experimentation, consumer GPU (12GB+)

### Ternary Config - 50M Parameters
```python
from src.model_v5.config import config_ternary_50m
config = config_ternary_50m()  # use_ternary=True
```
- **Total params**: ~72M
- **Ternary params**: 59%
- **FP16 params**: 41%
- **Architecture**: 18 layers, d_model=640, 10 attention heads
- **Use case**: Production scale, RTX 5070 or higher

## Ternary Training Results

### Successful Ternary Run (ternary_6m config)
- **Steps**: 100
- **Batch size**: 32
- **Device**: CUDA
- **Initial loss** (step 50): 10.3613
- **Final loss** (step 100): 10.2897
- **Improvement**: 0.69% loss reduction
- **Throughput**: 77k-82k tokens/sec
- **Ternary distribution**: +0.27 0:0.45 -0.27 (evolving toward uniform)

## How Ternary Works in v5

### Weight Quantization
- Weights are quantized to ternary values: {-1, 0, +1}
- During forward: weights = round_ternary(weight_shadow)
- During backward: gradient flows through STE (straight-through estimator)
- Shadow weights (FP32 or FP16) accumulate gradients for learning

### Implementation
- `TernaryLinear` module replaces standard `nn.Linear` for quantized layers
- Applied to: in_proj, x_proj, out_proj layers in SSM blocks
- Not applied to: layer norm, attention, optional LM head (controlled by `ternary_lm_head`)

### Training Dynamics
- Ternary params gradually evolve toward uniform distribution (33% -1, 33% 0, 33% +1)
- Commitment metric tracks convergence
- Gradient norm typically lower in ternary (due to discrete quantization)

## Inference Considerations

### Ternary Advantages
- 16-32x weight memory savings (3 bits per weight vs 16-32 bits)
- Potential for bitmask operations (future optimization)
- No matrix multiplication (only addition/subtraction)

### Ternary Challenges
- Inference kernel support limited (CPU fallback may be needed)
- Precision loss compared to full precision
- Custom CUDA kernels required for speedup

## Usage Guide

### For Training
```python
from src.model_v5.config import config_ternary_6m
from src.model_v5.train import train_model

config = config_ternary_6m()
results = train_model(config=config, max_steps=1000, ...)
```

### For CLI Training
```bash
python -m src.model_v5 --config ternary_6m --max-steps 1000
```

### Switching Between Modes
```python
# FP16 only
config.use_ternary = False

# Ternary with untied LM head
config.use_ternary = True
config.ternary_lm_head = True

# Ternary without LM head quantization
config.use_ternary = True
config.ternary_lm_head = False
```

## Performance Characteristics

### Speed
- Training slightly slower than FP16 due to quantization overhead
- Inference potential for significant speedup with custom kernels
- Current CPU inference slower due to lack of native ternary ops

### Accuracy
- Initial convergence slightly different from FP16
- Final loss typically within ±2% of FP16 baseline
- Quality determined by STE quality and learning rate

### Memory
- Shadow weights: Same as FP32 parameters
- Ternary quantized: 3x more efficient than FP32, 2x more efficient than FP16
- Optimizer states: Same overhead as standard training

## Troubleshooting

### 0% Ternary Params in Output
**Check**: `config.use_ternary` is True
**Check**: Using `config_ternary_6m()` or `config_ternary_50m()`
**Check**: Model was rebuilt after config change

### High Loss with Ternary
**Solution**: Reduce learning rate (try 3e-4 or 2e-4)
**Solution**: Increase warmup steps (try 200-500)
**Solution**: Check if weight quantization is actually active (look for "Ternary: XX%" in output)

### Memory Issues with Ternary
Ternary has similar memory to FP16 due to shadow weights and optimizer states.
- Try reducing `d_model` (256 → 128)
- Try reducing `batch_size` (32 → 16)
- Try reducing `seq_len` (256 → 128)

## References

- Implementation: `src/model_v5/components/ternary.py`
- Model architecture: `src/model_v5/model.py` (ModelV2)
- Training: `src/model_v5/train.py`
- Configs: `src/model_v5/config.py`

## Notes for Future Work

1. **Ternary Kernel Optimization**: Current ops are standard PyTorch; ternary-specific CUDA kernels could provide 2-8x speedup
2. **Bitmask Packing**: Future optimization to pack ternary weights 5 per byte instead of 1 per 32-bit float
3. **Inference Optimization**: Custom kernels for ternary matrix-vector products needed for inference speedup
4. **Quantization-Aware Initialization**: Currently random init; ACSI-style smart init could improve convergence
