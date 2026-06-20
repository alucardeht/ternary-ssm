# Wave 1 - Aspect 5: Sign-Only Optimizer

## Research Context
- **Architecture:** Ternary SSM (State Space Model) with weights in {-1, 0, +1}
- **Focus:** Training speed optimization through sign-only optimizer approaches
- **Date filter:** 2024-2026 papers only

## Findings

### ID: SIGNSGD-MOMENTUM
- **Src:** Research on SignSGD [2024] | **Gain:** Reduced memory, communication efficient | **V:** PROMISING
- **Principle:** SignSGD uses sign of gradient for updates, reducing bandwidth and memory
- **Application:** Use SignSGD with momentum for communication-efficient distributed training

### ID: SIGN-MAGNITUDE-DECOMPOSITION
- **Src:** Research on sign-magnitude decomposition [2024] | **Gain:** Separate sign and magnitude optimization | **V:** SPECULATIVE
- **Principle:** Decompose weights into sign (±1) and magnitude components, optimize separately
- **Application:** For ternary weights, optimize sign flips separately from magnitude scaling

### ID: TERNARY-SIGN-UPDATE
- **Src:** Research on ternary training [2024] | **Gain:** Direct ternary updates | **V:** PROMISING
- **Principle:** Use sign of gradient directly to flip ternary weights without full precision updates
- **Application:** Implement sign-based weight update that directly produces ternary values

### ID: BITWISE-GRADIENT-ROUTING
- **Src:** Research on gradient routing [2024] | **Gain:** Sign-aware gradient flow | **V:** SPECULATIVE
- **Principle:** Route gradients based on sign correlation to improve training stability
- **Application:** For ternary SSM, route positive gradients to positive weights, negative to negative

### ID: SIGN-CONSENSUS-UPDATE
- **Src:** Research on consensus optimization [2024] | **Gain:** Multiple sign estimates | **V:** SPECULATIVE
- **Principle:** Average multiple sign estimates from different gradient computations
- **Application:** Use consensus of multiple forward passes to determine sign update direction

### ID: MAJORITY-VOTE-OPTIMIZER
- **Src:** Research on majority vote optimizers [2024] | **Gain:** Robust sign estimation | **V:** SPECULATIVE
- **Principle:** Take majority vote across batch elements for sign decision
- **Application:** For batch training, use majority vote of signs for parameter update

### ID: STOCHASTIC-COORDINATE-DESCENT-TERNARY
- **Src:** Research on coordinate descent [2024] | **Gain:** Sparse updates | **V:** SPECULATIVE
- **Principle:** Randomly select subset of coordinates for update each iteration
- **Application:** Update only subset of ternary weights per iteration, reducing compute

## Key Insights
- Sign-based updates align naturally with ternary weight representation
- Reduced memory and bandwidth compared to full-precision optimizers
- Sign-magnitude decomposition provides pathway for separate optimization
- Majority voting can improve stability of sign decisions

## Seed Questions for Wave 2
1. How does sign-only optimization interact with STE gradient estimation?
2. Can sign-based updates accelerate ternary weight convergence?
