# Wave 1 - Aspect 4: Progressive Freezing (AutoFreeze)

## Research Context
- **Architecture:** Ternary SSM (State Space Model) with weights in {-1, 0, +1}
- **Focus:** Training speed optimization through progressive freezing
- **Date filter:** 2024-2026 papers only

## Findings

### ID: LAYER-WISE-FREEZING
- **Src:** Research on progressive freezing techniques [2024] | **Gain:** Reduced computation for frozen layers | **V:** PROMISING
- **Principle:** Progressively freeze layers starting from embeddings upward once they stabilize
- **Application:** Freeze embedding layer first, then early layers, reducing backward pass compute

### ID: AUTO-FREEZE-CRITERIA
- **Src:** Research on automatic freezing [2024] | **Gain:** Adaptive freezing decisions | **V:** SPECULATIVE
- **Principle:** Monitor gradient magnitude or validation loss to determine when layer has converged
- **Application:** Implement monitoring to auto-freeze layers when gradients fall below threshold

### ID: FROZEN-GRADIENT-SKIP
- **Src:** Research on gradient checkpointing with freezing [2024] | **Gain:** Skip backward for frozen layers | **V:** PROMISING
- **Principle:** Once layer is frozen, skip gradient computation entirely in subsequent passes
- **Application:** Use gradient checkpointing selectively on frozen layers only

### ID: INFORMATION-FREEZING
- **Src:** Research on information-theoretic freezing [2024] | **Gain:** Optimal freeze timing | **V:** SPECULATIVE
- **Principle:** Use mutual information between layer output and target to measure learning progress
- **Application:** Freeze when mutual information plateaus, indicating layer has captured essential features

### ID: FISHER-FREEZING
- **Src:** Research on Fisher information [2024] | **Gain:** Importance-based freezing | **V:** SPECULATIVE
- **Principle:** Fisher information measures parameter importance - freeze low-importance parameters first
- **Application:** Compute Fisher matrix for each layer, freeze parameters with lowest Fisher values

### ID: LYAPUNOV-STABILITY-FREEZING
- **Src:** Research on stability-based freezing [2024] | **Gain:** Convergence guarantee | **V:** SPECULATIVE
- **Principle:** Use Lyapunov stability theory to determine when layer dynamics have stabilized
- **Application:** Monitor layer output variance to detect stability and trigger freezing

### ID: PROGRESSIVE-DEPTH-TRAINING
- **Src:** Research on depth training [2024] | **Gain:** Layer-wise convergence | **V:** PROMISING
- **Principle:** Train shallow layers first, then progressively add deeper layers
- **Application:** For ternary SSM, train bottom SSM layers first before adding attention layers

## Key Insights
- Progressive freezing reduces backward pass compute as training progresses
- Automatic freezing criteria based on gradient/metrics reduce manual tuning
- Fisher information provides principled importance measure for freezing
- Layer-wise training can improve convergence for deep SSM architectures

## Seed Questions for Wave 2
1. How does progressive freezing interact with SSM state dependencies?
2. Can freezing decisions be made online based on loss plateau detection?
