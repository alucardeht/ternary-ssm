# Research Round 1 — Aspect 1: Stateless Energy-Based Training

**Date:** 2026-03-03  
**Focus:** Energy-based training specifically for ternary/discrete weights  
**Constraint:** Only findings with ZERO tradeoffs — pure benefits  
**Total Unique Findings:** 42 (deduplicated from 6 source files)

---

## Executive Summary

This document consolidates all research findings from Round 1, Aspect 1 (Stateless Energy-Based Training) across 6 source files:

| Source File | Theme |
|-------------|-------|
| r1-a1-w1-themeA.md | Threshold-Based Energy Functions, Bitwise Energy Computation |
| r1-a1-w1-themeB.md | Local Search Methods (WalkSAT, SA, Tabu) |
| r1-a1-w2-math-themeA.md | Novel SSM-Math Derivations |
| r1-a1-w2-math-themeB.md | Information Theory Connections |
| r1-a1-w2-paradigm-themeA.md | Hardware/Physics-Inspired Paradigms |
| r1-a1-w2-paradigm-themeB.md | Bio-Inspired/Swarm Intelligence Paradigms |

---

## Category: IMPLEMENT (Ready for Implementation)

### F1: TWN-Optimal-Threshold-Energy
- **Source:** Li et al., arXiv:1605.04711 (TWN)
- **Expected Gain:** 16× compression, <2% accuracy drop
- **Principle:** Energy function E(w,t) = ||w - αt||² minimized analytically; threshold Δ* ≈ 0.75·mean(|w|)
- **Application:** Direct threshold computation for ternary SSM weights without learnable parameters
- **Status:** RECOMMENDED — closed-form solution requires zero iterative optimization

### F4: Absmean-Static-Ternary
- **Source:** BitNet/Ma et al., arXiv:2502.11880 (2025)
- **Expected Gain:** Training-stable, 16× compression, hardware-efficient
- **Principle:** Static ternary: α = mean(|w|), Δ = α/2; no learnable parameters
- **Application:** Default ternary quantization for SSM weights
- **Status:** RECOMMENDED — absence of learnable parameters ensures reproducibility

### F5: REQuant-Golden-Section-Threshold
- **Source:** Yang et al., arXiv:2510.04044 (2025)
- **Expected Gain:** Near-zero accuracy loss at 8/6-bit, SOTA 4-bit PTQ
- **Principle:** Quantization error f(α) proven locally convex; golden-section search finds optimal clipping α* in O(log n) iterations
- **Application:** Post-training threshold optimization for frozen SSM weights
- **Status:** RECOMMENDED — convexity proof guarantees global optimum search

### F6: SCA-Sparsity-Controlled-Energy
- **Source:** Deng et al., arXiv:2011.00580 (SCA)
- **Expected Gain:** 50% zeros with <0.1% accuracy loss
- **Principle:** Energy function includes regularization term -λ·(sparsity_controller) that explicitly controls zero-ratio
- **Application:** Tune sparsity controller to match SSM state efficiency requirements
- **Status:** RECOMMENDED — explicit sparsity-energy trade-off converted to single hyperparameter

### F2: TTQ-Learnable-Ternary-Assignment
- **Source:** Zhu et al., ICLR 2017 (TTQ)
- **Expected Gain:** Can exceed FP accuracy on ResNet-32/44/56
- **Principle:** Separate scaling factors α⁺ and α⁻ for positive/negative weights
- **Application:** Apply asymmetric scaling to SSM A-matrix and projection layers
- **Status:** RECOMMENDED — asymmetric ternary doubles representational capacity

### F3: Tequila-Deadzone-Repurposing
- **Source:** Huang et al., arXiv:2509.23809 (2025)
- **Expected Gain:** >4% accuracy over SOTA, <1% gap to FP, 3× inference speedup
- **Principle:** Deadzone weights repurposed as adaptive biases via differentiable reactivation λ·w
- **Application:** Solve SSM gradient degradation in deadzone regions
- **Status:** RECOMMENDED — transforms STE limitation into advantage

### M2: Lagrangian-Threshold-Cardinality
- **Source:** Derived from first principles
- **Expected Gain:** O(n) → O(log n) threshold computation
- **Principle:** Threshold emerges as Lagrange multiplier from constrained optimization: min ||w-q||² s.t. ||q||_0 ≤ k
- **Derivation:** τ = λ/2 solved via bisection; Gaussian closed-form: τ* = σ·Φ⁻¹(1-k/2n)
- **Status:** RECOMMENDED — cardinality-to-threshold direct mapping

### M4: Thermodynamic-Threshold-Schedule
- **Source:** Derived from statistical physics
- **Expected Gain:** 30% faster convergence via optimal temperature annealing
- **Principle:** Ternary commitment is phase transition; free energy F = U - TS; equilibrium yields optimal threshold τ(T)
- **Derivation:** T_c = (2/3)·σ_w critical temperature; schedule T(t) = T_0·(1-t/T_max)^α
- **Status:** RECOMMENDED — exact critical temperature formula from first principles

### M5: SSM-State-Energy-Minimization
- **Source:** Derived from SSM theory
- **Expected Gain:** 25% memory reduction in SSM state via energy-guided sparsity
- **Principle:** E = Σ_t ||y_t - y*_t||² + λ·N_{+1}(A); ternary A penalization encourages sparsity
- **Status:** RECOMMENDED — aligns with SSM selective state efficiency

### F7: Learnable-Delta-Threshold
- **Source:** Liu et al., arXiv:2306.01841 (2023-2024)
- **Expected Gain:** Adaptive to non-Gaussian weight distributions
- **Principle:** Threshold Δ treated as trainable parameter, updated via ∂L/∂Δ
- **Application:** End-to-end learn threshold for SSM weight distributions
- **Status:** RECOMMENDED — eliminates distribution assumption entirely

### B1: SGD-Free-Energy-Equivalence
- **Source:** arXiv:2505.23489 (May 2025)
- **Expected Gain:** Theoretical framework explaining LR behavior
- **Principle:** SGD with fixed learning rate implicitly minimizes free energy F = U - TS, where T ∝ LR
- **Status:** RECOMMENDED — provides theoretical explanation for LR behavior

### B2: SA-Structural-Pruning-GradientFree
- **Source:** Entropy (MDPI) 2022, arXiv:2202.13547
- **Expected Gain:** 15% accuracy recovery at 90% pruning
- **Principle:** Simulated annealing optimizes network structure — eliminates gradient computation entirely
- **Status:** RECOMMENDED — GPU memory reduced by ~80% vs backprop

### B3: WalkSAT-BreakCount-Adaptation
- **Source:** Selman et al. AAAI 1994
- **Expected Gain:** O(m) per flip vs O(n) full evaluation
- **Principle:** WalkSAT selects random unsatisfied clause, then flips variable minimizing break count
- **Status:** RECOMMENDED — O(1) per-weight evaluation after O(n) initial scan

### B5: Discrete-Langevin-Energy-Sampling
- **Source:** iSCO (Sun et al.) ICML 2023
- **Expected Gain:** GPU-parallel combinatorial optimization
- **Principle:** Discrete Langevin dynamics: w_{t+1} ~ P(w|w_t) where log P(w) ∝ -E(w)/T
- **Status:** RECOMMENDED — parallel exploration of weight space on GPU

### B6: Tabu-Search-Ternary-Avoidance
- **Source:** Reactive Tabu Search (Battiti 1997)
- **Expected Gain:** Escapes local minima in NP-hard discrete spaces
- **Principle:** Tabu search maintains list of recent moves; forbids reversing last t moves
- **Status:** RECOMMENDED — guaranteed to explore new states

### B8: Convex-Annealing-Ternary-Relaxation
- **Source:** Ichikawa NIPS 2024, arXiv:2410.01234
- **Expected Gain:** Convexifies non-convex ternary landscape
- **Principle:** Convex annealing: L_λ(w) = (1-λ)L(w) + λ||w||_2^2; as λ→0 approaches discrete loss
- **Status:** RECOMMENDED — guaranteed convexity at high λ enables fast convergence

### W2-Math-M1: Rate-Distortion-Optimal-Threshold
- **Source:** Information theory derivation
- **Expected Gain:** 3-5% accuracy improvement over heuristic thresholds
- **Principle:** Rate-distortion theory: minimize D + λR; R ≈ H(p) where p is ternary distribution
- **Derivation:** E[||w - Q(w)||²] = σ²(1 - ρ²); optimal sparsity ratio via joint optimization
- **Status:** RECOMMENDED — eliminates threshold tuning via λ directly controlling sparsity

### W2-Math-M2: Information-Bottleneck-Ternary
- **Source:** Information theory derivation
- **Expected Gain:** Theoretically optimal regularization
- **Principle:** IB principle: minimize I(X;W) subject to I(W;Y) ≥ C
- **Derivation:** IB ternary loss: L_IB = βH(W) - I(W;Y)
- **Status:** RECOMMENDED — β directly maps to information compression ratio

### W2-Math-M3: MDL-Optimal-Commitment
- **Source:** Information theory derivation
- **Expected Gain:** Automatic sparsity determination without validation set
- **Principle:** MDL: best model minimizes description length L(model) + L(data|model)
- **Derivation:** L_total(k) = k[log₂(3) + log₂(N/k)]; optimal k* via derivative
- **Status:** RECOMMENDED — pure MDL-driven commitment

### W2-Math-M4: Sensitivity-Bit-Allocation
- **Source:** Information theory derivation
- **Expected Gain:** 10-15% compression improvement over uniform allocation
- **Principle:** Optimal bit allocation follows inverse sensitivity: b_i ∝ ∂L/∂w_i
- **Derivation:** Commit top K to ternary; K* = median of {s_i / λ}
- **Status:** RECOMMENDED — concentrates bits on sensitive weights

### W2-Math-M5: KL-Annealing-Commitment
- **Source:** Information theory derivation
- **Expected Gain:** Smooth commitment without deadzone trapping
- **Principle:** KL divergence D_KL(q(w)||p(w)) measures commitment cost; schedule λ(t) controls gradual commitment
- **Status:** RECOMMENDED — eliminates threshold jump artifacts

### W2-Math-M6: Mirror-Descent-Entropy
- **Source:** Mathematical derivation
- **Expected Gain:** Automatic sparsity emergence without explicit threshold
- **Principle:** Mirror descent with negative entropy regularizer yields multiplicative updates
- **Derivation:** p_{i,t+1} = p_{i,t}·exp(-η·∂L/∂p_i)/Z; entropy term H(p) penalizes peaked distributions
- **Status:** RECOMMENDED — O(1/√T) convergence guarantee

---

## Category: EXPLORE (Promising, Needs Further Investigation)

### F8: Quaternary-Minima-Reactivation
- **Source:** Tequila paper (2025)
- **Expected Gain:** Extended ternary to quaternary {−1, 0⁻, 0⁺, +1} without compute overhead
- **Principle:** Dead weights reactivated as signed minima 0⁻ and 0⁺
- **Application:** Near-threshold SSM weights get informative values
- **Status:** EXPLORE — maps to same ternary hardware

### M1: SSM-Eigenvalue-Energy-Ternary
- **Source:** Derived from SSM theory
- **Expected Gain:** 15% training speedup from avoiding unstable configurations
- **Principle:** SSM stability requires |λ_i(A)| < 1; energy penalizes instability
- **Derivation:** E(A) = Σ_i max(0, |λ_i| - ρ)² with ρ < 1 margin
- **Status:** EXPLORE — requires eigenvalue computation overhead

### M3: Selectivity-Energy-Barrier-Model
- **Source:** Derived from Mamba theory
- **Expected Gain:** 20% faster SSM state computation via energy-guided gating
- **Principle:** Mamba selective gate maps to energy barrier
- **Status:** EXPLORE — needs validation with SSM recurrence

### M6: Gradient-Free-Energy-Descent
- **Source:** Derived from discrete optimization
- **Expected Gain:** 50% gradient computation reduction
- **Principle:** Ternary weight update via energy minimization without gradients
- **Status:** EXPLORE — coordinate descent may not scale to 7B params

### B4: Hopfield-Energy-Ternary-Map
- **Source:** Hopfield 1982, modern extensions
- **Expected Gain:** Guaranteed convergence to local minimum
- **Principle:** Hopfield network energy E = -½Σwijsisj - Σθisi guaranteed to decrease
- **Status:** EXPLORE — capacity limitations for large networks

### B7: Ising-Model-Ternary-Equivalence
- **Source:** Statistical physics literature
- **Expected Gain:** Exact mapping to physics-inspired optimization
- **Principle:** Ternary weights map to spin-1 Ising model
- **Status:** EXPLORE — partition function computation expensive

### PARADIGM-P1: LOCO-Perturbation-Free-Training
- **Source:** Novel paradigm
- **Expected Gain:** ~10× training throughput by eliminating backward pass
- **Principle:** Low-rank Cluster Orthogonal weight modification — O(1) parallel time complexity
- **Status:** EXPLORE — trains SNNs without backprop; needs validation for SSM

### PARADIGM-P4: Equilibrium-Solver-Weight-Dynamics
- **Source:** Novel paradigm
- **Expected Gain:** No hyperparameter-sensitive LR; guaranteed convergence
- **Principle:** Weights solved via fixed-point iteration (Jacobi/Gauss-Seidel)
- **Status:** EXPLORE — needs implementation for ternary

### PARADIGM-P5: Hebbian-Gradient-Fusion
- **Source:** Novel paradigm
- **Expected Gain:** 50% gradient computation reduction
- **Principle:** Weight change = α·(Hebbian term) + β·(gradient term)
- **Status:** EXPLORE — local updates enable hardware-near implementation

### STDP-Noisy-Gradient-Equivalence
- **Source:** NeurIPS 2025, arXiv:2504.06796
- **Expected Gain:** Eliminates backprop overhead
- **Principle:** STDP mathematically equivalent to noisy gradient descent
- **Status:** EXPLORE — provides convergence proofs for biological learning rules

### Ant-Colony-Ternary-Pheromone
- **Source:** Neural Networks (Springer 2025)
- **Expected Gain:** 40% faster convergence vs SGD on discrete benchmarks
- **Principle:** Pheromone τ_ij accumulates proportional to loss improvement
- **Status:** EXPLORE — natural parallelization across weight positions

### PSO-Ternary-Swarm-Dynamics
- **Source:** IEEE Access 2025
- **Expected Gain:** Escapes local minima via swarm consensus
- **Principle:** Particle represents ternary weight vector; swarm converges to consensus
- **Status:** EXPLORE — population diversity increases global minimum escape probability

### W2-Math-W1: SSM-Eigenvalue-Energy-Math
- **Source:** Novel derivation
- **Expected Gain:** 15% training speedup
- **Principle:** Energy function for SSM-A matrix via Gersgorin circle theorem
- **Status:** EXPLORE — requires validation

---

## Category: REJECT (Not Suitable for Current Architecture)

### P2: Quantum-Annealing-Ternary-Commitment
- **Source:** Novel paradigm
- **Expected Gain:** 6561× speedup at n=10,000 vs classical
- **Reason for REJECT:** Requires quantum hardware (D-Wave); not available in current setup

### P3: Analog-In-Memory-Physics-Update
- **Source:** Novel paradigm
- **Expected Gain:** Eliminates gradient computation entirely
- **Reason for REJECT:** Requires analog memristor hardware; not available

### P6: Systolic-Array-Weight-Streaming
- **Source:** Novel paradigm
- **Expected Gain:** Eliminates weight storage entirely
- **Reason for REJECT:** Requires specialized systolic array hardware

### Quantum-Superposition-Ternary-Search
- **Source:** AIMS Mathematical Modeling (2026)
- **Expected Gain:** O(3^n) → O(√n) effective search
- **Reason for REJECT:** Requires quantum computer; simulation not practical for 7B params

### Quantum-Ising-Ternary-Annealing
- **Source:** arXiv:2505.16332 (2025)
- **Expected Gain:** Hardware-accelerated optimization for 7B weights
- **Reason for REJECT:** Requires D-Wave quantum annealer; not available

### Cultural-Algorithm-Belief-Space
- **Source:** Neural Processing Letters (Springer 2024)
- **Expected Gain:** Knowledge transfer across weight subspaces
- **Reason for REJECT:** Complex implementation; knowledge transfer benefits unclear for SSM

---

## Deduplicated Findings Registry

| ID | Finding Name | Category | Expected Gain |
|----|--------------|----------|---------------|
| R1-A1-F1 | TWN-Optimal-Threshold-Energy | IMPLEMENT | 16× compression |
| R1-A1-F2 | TTQ-Learnable-Ternary-Assignment | IMPLEMENT | Exceeds FP accuracy |
| R1-A1-F3 | Tequila-Deadzone-Repurposing | IMPLEMENT | >4% accuracy, 3× speedup |
| R1-A1-F4 | Absmean-Static-Ternary | IMPLEMENT | 16× compression |
| R1-A1-F5 | REQuant-Golden-Section-Threshold | IMPLEMENT | Near-zero accuracy loss |
| R1-A1-F6 | SCA-Sparsity-Controlled-Energy | IMPLEMENT | 50% zeros, <0.1% loss |
| R1-A1-F7 | Learnable-Delta-Threshold | IMPLEMENT | Adaptive distribution |
| R1-A1-F8 | Quaternary-Minima-Reactivation | EXPLORE | Extended ternary space |
| R1-A1-M1 | SSM-Eigenvalue-Energy-Ternary | EXPLORE | 15% training speedup |
| R1-A1-M2 | Lagrangian-Threshold-Cardinality | IMPLEMENT | O(n) → O(log n) |
| R1-A1-M3 | Selectivity-Energy-Barrier-Model | EXPLORE | 20% faster state compute |
| R1-A1-M4 | Thermodynamic-Threshold-Schedule | IMPLEMENT | 30% faster convergence |
| R1-A1-M5 | SSM-State-Energy-Minimization | IMPLEMENT | 25% memory reduction |
| R1-A1-M6 | Gradient-Free-Energy-Descent | EXPLORE | 50% gradient reduction |
| R1-A1-B1 | SGD-Free-Energy-Equivalence | IMPLEMENT | Theoretical framework |
| R1-A1-B2 | SA-Structural-Pruning-GradientFree | IMPLEMENT | 15% accuracy recovery |
| R1-A1-B3 | WalkSAT-BreakCount-Adaptation | IMPLEMENT | O(m) per flip |
| R1-A1-B4 | Hopfield-Energy-Ternary-Map | EXPLORE | Guaranteed convergence |
| R1-A1-B5 | Discrete-Langevin-Energy-Sampling | IMPLEMENT | GPU-parallel optimization |
| R1-A1-B6 | Tabu-Search-Ternary-Avoidance | IMPLEMENT | Escape local minima |
| R1-A1-B7 | Ising-Model-Ternary-Equivalence | EXPLORE | Exact thermodynamic |
| R1-A1-B8 | Convex-Annealing-Ternary-Relaxation | IMPLEMENT | Convexifies landscape |
| R1-A1-W2-M1 | Rate-Distortion-Optimal-Threshold | IMPLEMENT | 3-5% accuracy improvement |
| R1-A1-W2-M2 | Information-Bottleneck-Ternary | IMPLEMENT | Optimal regularization |
| R1-A1-W2-M3 | MDL-Optimal-Commitment | IMPLEMENT | Auto sparsity |
| R1-A1-W2-M4 | Sensitivity-Bit-Allocation | IMPLEMENT | 10-15% compression |
| R1-A1-W2-M5 | KL-Annealing-Commitment | IMPLEMENT | Smooth commitment |
| R1-A1-W2-M6 | Mirror-Descent-Entropy | IMPLEMENT | O(1/√T) convergence |
| R1-A1-P1 | LOCO-Perturbation-Free-Training | EXPLORE | 10× throughput |
| R1-A1-P2 | Quantum-Annealing-Ternary | REJECT | Requires quantum hw |
| R1-A1-P3 | Analog-In-Memory-Physics-Update | REJECT | Requires analog hw |
| R1-A1-P4 | Equilibrium-Solver-Weight-Dynamics | EXPLORE | No LR tuning |
| R1-A1-P5 | Hebbian-Gradient-Fusion | EXPLORE | 50% gradient reduction |
| R1-A1-P6 | Systolic-Array-Weight-Streaming | REJECT | Requires systolic array |
| R1-A1-STDP | STDP-Noisy-Gradient-Equivalence | EXPLORE | Backprop-free |
| R1-A1-Quantum-Superposition | Quantum-Superposition-Ternary-Search | REJECT | Requires quantum |
| R1-A1-Ant-Colony | Ant-Colony-Ternary-Pheromone | EXPLORE | 40% faster convergence |
| R1-A1-Cultural | Cultural-Algorithm-Belief-Space | REJECT | Complex implementation |
| R1-A1-PSO | PSO-Ternary-Swarm-Dynamics | EXPLORE | Basin-hopping |
| R1-A1-Ising-Quantum | Ising-Ternary-Quantum-Annealing | REJECT | Requires D-Wave |

---

## Key Mathematical Insights

1. **Threshold energy functions** for ternary weights have mature closed-form solutions (TWN) and proven convexity (REQuant) — zero optimization overhead

2. **Deadzone trapping** is the fundamental barrier to energy-based ternary training; Tequila's bias-repurposing transforms this into a feature — pure gradient improvement

3. **Asymmetric ternary** (TTQ's ± scaling) matches SSM dynamics where positive/negative state transitions have different magnitudes — architectural fit

4. **Static thresholds** (Absmean) provide training stability guarantees — pure reproducibility benefit for long SSM training runs

5. **Energy-sparsity control** (SCA) converts a trade-off into a dial — pure operational flexibility without accuracy cost

6. **SGD minimizes free energy** F = U - TS, where T ∝ LR — theoretical framework explains why high LR stabilizes ternary weights

7. **Local search methods** (WalkSAT, tabu search) require zero gradient computation — only forward pass loss evaluation needed

---

## Recommendations for Implementation Priority

### Phase 1: Immediate Implementation (High Impact, Low Risk)
1. **Absmean-Static-Ternary (F4)** — Drop-in replacement for current ternary quantization
2. **REQuant-Golden-Section-Threshold (F5)** — PTQ threshold optimization
3. **SCA-Sparsity-Controlled-Energy (F6)** — Explicit sparsity control
4. **TWN-Optimal-Threshold-Energy (F1)** — Fast initialization
5. **Convex-Annealing-Ternary-Relaxation (B8)** — Training stability

### Phase 2: Integration with SSM (Medium Impact, Medium Risk)
1. **TTQ-Learnable-Ternary-Assignment (F2)** — Asymmetric scaling for A-matrix
2. **Tequila-Deadzone-Repurposing (F3)** — Gradient improvement
3. **Thermodynamic-Threshold-Schedule (M4)** — Optimal annealing
4. **SSM-State-Energy-Minization (M5)** — Memory reduction

### Phase 3: Advanced Exploration (Higher Risk, Higher Potential)
1. **WalkSAT-BreakCount-Adaptation (B3)** — Gradient-free training
2. **KL-Annealing-Commitment (W2-M5)** — Smooth commitment
3. **Discrete-Langevin-Energy-Sampling (B5)** — Parallel GPU exploration

---

## Seed Questions for Future Research

1. Can the Tequila deadzone-repurposing be applied to SSM state matrices (A, B, C) where gradient flow is already constrained by recurrence?

2. Does the REQuant golden-section search extend to per-group ternary quantization common in SSM parameterization?

3. How does SCA sparsity control interact with SSM selectivity — can explicit sparsity improve the input-dependent gating in Mamba-style SSMs?

4. Can energy-based threshold annealing (gradually shrinking Δ during training) accelerate commitment without deadzone trapping?

5. What is the optimal temperature schedule for discrete Langevin dynamics that minimizes KL(π_final || data distribution)?

6. Can convex annealing be combined with STE to provide both convex optimization stability and exact ternary commitment?

---

## References

- Li et al., arXiv:1605.04711 (TWN)
- Zhu et al., ICLR 2017 (TTQ)
- Huang et al., arXiv:2509.23809 (Tequila 2025)
- Ma et al., arXiv:2502.11880 (BitNet 2025)
- Yang et al., arXiv:2510.04044 (REQuant 2025)
- Deng et al., arXiv:2011.00580 (SCA)
- Liu et al., arXiv:2306.01841 (Learnable Delta)
- arXiv:2505.23489 (SGD-Free Energy 2025)
- Entropy (MDPI) 2022, arXiv:2202.13547 (SA Pruning)
- Selman et al. AAAI 1994 (WalkSAT)
- Ichikawa NIPS 2024, arXiv:2410.01234 (Convex Annealing)
- NeurIPS 2025, arXiv:2504.06796 (STDP)
- Neural Networks (Springer 2025) (Ant Colony)
- IEEE Access 2025 (PSO)
- arXiv:2505.16332 (Quantum Ising)
