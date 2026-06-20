# Model Composition: TIES Merging for Ternary Models

This module implements **TIES (Task-Integrated Embedding Space)** merging, adapted specifically for ternary weight models.

## Overview

TIES merging enables combining multiple trained models into a single merged model that preserves strengths from all inputs. Unlike simple averaging, TIES intelligently handles conflicting weights through:

1. **Sign Voting**: For ternary weights {-1, 0, +1}, count agreement across models
2. **Confidence Scoring**: Identify high-confidence vs conflicting weights
3. **Trimming**: Remove low-confidence weights to reduce interference
4. **Weighted Averaging**: Merge remaining weights with per-model importance weights

## Use Cases

### 1. Multi-Task Model Merging
Train separate ternary models on different tasks, then merge into a single generalist model:
```python
# Task 1: Trained on code completion
model_code = ModelV2(config)
# ... train on code corpus ...

# Task 2: Trained on math problems
model_math = ModelV2(config)
# ... train on math corpus ...

# Merge both into generalist
merger = TernaryTIESMerger()
merged_state = merger.merge([model_code, model_math])

generalist = ModelV2(config)
generalist.load_state_dict(merged_state)
```

### 2. Ensemble Merging
Combine multiple fine-tuned variants into one efficient model:
```python
# Fine-tune same base model on different data splits
models = []
for i in range(3):
    model = ModelV2(config)
    # ... fine-tune on data_split_i ...
    models.append(model)

# Merge all variants
merger = TernaryTIESMerger(trim_threshold=0.15)
merged = merger.merge(models)
```

### 3. Weighted Task Merging
Prioritize certain tasks over others:
```python
# Code is 3x more important than math
task_weights = {'code': 3.0, 'math': 1.0}

merger = TernaryTIESMerger(tie_weights=[3.0, 1.0])
merged_state = merger.merge([model_code, model_math])
```

## API Reference

### TernaryTIESMerger

Main class for merging models.

#### Constructor
```python
TernaryTIESMerger(
    trim_threshold: float = 0.15,
    scaling_factor: float = 1.0,
    tie_weights: Optional[List[float]] = None,
    device: str = 'cpu',
)
```

**Parameters:**
- `trim_threshold`: Fraction of low-confidence weights to trim (0.15 = 15%). Higher values = more aggressive trimming
- `scaling_factor`: Multiplier for task-specific parameters (typically 1.0)
- `tie_weights`: Per-model weights for importance. If None, use uniform weights (1/N each)
- `device`: Device for merge computation ('cpu' or 'cuda')

#### Methods

##### `merge(models: List[nn.Module]) -> Dict[str, torch.Tensor]`
Merge multiple models using TIES algorithm.

**Args:**
- `models`: List of trained models with identical architecture

**Returns:**
- State dict ready to load into new model

**Example:**
```python
merger = TernaryTIESMerger()
merged_state = merger.merge([model1, model2, model3])

new_model = ModelV2(config)
new_model.load_state_dict(merged_state)
```

##### `get_merge_stats() -> Dict[str, int]`
Get statistics about the merge operation.

**Returns:**
- Dict with:
  - `ternary_params`: Number of ternary weights merged
  - `dense_params`: Number of dense (non-ternary) weights merged

**Example:**
```python
stats = merger.get_merge_stats()
print(f"Merged {stats['ternary_params']} ternary weights")
print(f"Merged {stats['dense_params']} dense weights")
```

### validate_merge_quality

Evaluate quality of a merged model.

```python
metrics = validate_merge_quality(
    original_models=[model1, model2, model3],
    merged_model=merged_model,
    test_input=test_tensor,
    device='cpu'
)
```

**Returns dict with:**
- `output_agreement`: How well merged matches average of originals (0-1, higher=better)
- `weight_divergence`: Average L2 divergence of merged weights from originals

## Implementation Details

### Ternary Weight Detection
Weights are identified as ternary if they:
1. Contain 'shadow' in parameter name (STE shadow weights), OR
2. Have ≤5 unique values (characteristic of {-1, 0, +1} quantized weights)

### Merging Algorithm

For **ternary weights**:
1. Extract sign of each weight in each model: `sign = torch.sign(w)`
2. Count sign agreement: `agreement = sum(signs) / n_models` ∈ [-1, 1]
3. Confidence score: `confidence = |agreement|` ∈ [0, 1]
4. Trim mask: `weight = 0 if confidence < trim_threshold`
5. Weighted average: `merged = Σ(weight_i * model_i.weight)`
6. Re-quantize: `merged = sign(merged)` → {-1, 0, +1}

For **dense weights**:
- Simple weighted average: `merged = Σ(weight_i * model_i.weight)`

### Hyperparameter Tuning

**trim_threshold (default 0.15):**
- `0.0`: No trimming, pure weighted average
- `0.15`: Standard (recommended for most cases)
- `0.3+`: Aggressive trimming, focuses on highly-agreed weights

**tie_weights:**
- All equal (default): `[1/N, 1/N, ..., 1/N]`
- Prioritize one model: `[0.5, 0.3, 0.2]`
- Strongly prioritize: `[0.8, 0.1, 0.1]`

## Testing

Run the test suite:
```bash
cd src/model_v5
python -m composition.tests
```

Tests include:
- Basic merge functionality
- Weighted merging
- Trim threshold variations
- Output quality validation

## Performance

### Memory
- Merge operation: O(N * M) where N = number of models, M = model size
- No intermediate copies needed if done on CPU
- ~2-3x model size temporarily during merge

### Speed
- CPU merge time: ~10-100ms for 100M parameter model
- GPU merge time: ~1-10ms for 100M parameter model
- Insignificant compared to model training time

## References

- **TIES-Merging Paper**: "TIES-Merging: Resolving Interference When Merging Models" (Xinghao et al., 2024)
- **Ternary Weights**: Adapted for {-1, 0, +1} quantization via sign voting
- **Multi-Task Learning**: Inspired by task-aware merging techniques

## Limitations & Future Work

### Current Limitations
1. Requires identical model architectures
2. No support for partial merging (all parameters merged)
3. No automatic hyperparameter tuning

### Future Enhancements
1. **Architecture-agnostic merging**: Support different layer configurations
2. **Selective merging**: Choose which layers/parameters to merge
3. **Adaptive trim threshold**: Automatically tune based on weight distribution
4. **LoRA compatibility**: Merge LoRA adapters separately from base weights
5. **Expert routing**: For MoE models, route tasks to appropriate experts

## Examples

### Example 1: Three-Model Merge
```python
from src.model_v5.config import ModelConfig
from src.model_v5.model import ModelV2
from src.model_v5.composition.ties_merge import TernaryTIESMerger

config = ModelConfig(d_model=256, n_layer=4, ...)

# Train three models on different corpora
model_a = ModelV2(config)
# ... train on corpus A ...

model_b = ModelV2(config)
# ... train on corpus B ...

model_c = ModelV2(config)
# ... train on corpus C ...

# Merge
merger = TernaryTIESMerger(trim_threshold=0.15)
merged_state = merger.merge([model_a, model_b, model_c])

# Load merged
final_model = ModelV2(config)
final_model.load_state_dict(merged_state)

# Use generalist model
output = final_model(input_ids)
```

### Example 2: Weighted Merge with Quality Check
```python
from src.model_v5.composition.ties_merge import validate_merge_quality

# Merge with custom weights
merger = TernaryTIESMerger(tie_weights=[0.4, 0.3, 0.3])
merged_state = merger.merge([model_a, model_b, model_c])

final_model = ModelV2(config)
final_model.load_state_dict(merged_state)

# Validate quality
test_input = torch.randint(0, 2048, (8, 512))
metrics = validate_merge_quality(
    [model_a, model_b, model_c],
    final_model,
    test_input
)

print(f"Output agreement: {metrics['output_agreement']:.4f}")
print(f"Weight divergence: {metrics['weight_divergence']:.6f}")
```

## Contact & Issues

For questions or issues with TIES merging, see the main project documentation or contact the development team.
