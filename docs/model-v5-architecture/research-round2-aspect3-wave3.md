# Research Round 2, Aspect 3: Convergence Speed — Wave 3 (Paradigm-Breakers)

## Summary

Two paradigm-breakers explored radical alternatives for faster ternary training convergence. After deduplication (EDTT ≈ FPQ merged), 9 unique paradigm proposals remain. The unifying insight: standard STE+Adam spends the vast majority of gradient steps on informationally empty updates — shadow weights move within ternary basins without changing any ternary value. Every proposal below concentrates compute on the MOMENTS OF CHANGE — threshold crossings where ternary values actually flip.

## TIER 1 — HIGH VIABILITY (test first)

### R2-A3-P1: ACSI — Analytical Construction via System Identification
- **Paradigm shift:** Do not TRAIN SSM weights. CONSTRUCT them analytically from data statistics using Ho-Kalman realization algorithm. Reduce training from 100K+ steps to a few thousand polishing steps.
- **Key insight:** SSM layer h_{t+1} = A h_t + B x_t is a LINEAR DYNAMICAL SYSTEM. System identification theory recovers (A, B, C, D) from input-output correlations via a single SVD of the Hankel matrix — O(n^3) cost, seconds not hours.
- **Method:** Compute Markov parameters h_k = E[y_{t+k} x_t^T] / E[x_t x_t^T] → build Hankel matrix → SVD → extract (A,B,C,D) → quantize to ternary → polish with short gradient training (1-5K steps).
- **Multi-layer:** Greedy layer-by-layer construction (output of layer L becomes input to construct layer L+1).
- **Convergence speedup:** 20-100x fewer training steps. Construction phase adds negligible time (seconds of SVD).
- **Feasibility:** HIGH for SSM layers (60+ years of system identification theory). MEDIUM for attention layers (need PMI-based initialization).
- **Synergies:** Strongest with EDTT (P4) — construction provides near-optimal init, EDTT polishes remaining misaligned weights. Combined: 50-200x theoretical speedup.

### R2-A3-P2: HSC — Hashed Shadow Weight Compression
- **Paradigm shift:** Replace N independent shadow weights with a MUCH SMALLER table of K shared shadow values (K = N/R). Each weight position maps to table entry via hash function.
- **Key insight:** Johnson-Lindenstrauss lemma guarantees random projections preserve distances. Hash function is a random projection from N to K dimensions. Gradient per table entry averages over R positions → noise reduced by sqrt(R).
- **Method:** shadow_i = s_1(i) * T[h_1(i)] + s_2(i) * T[h_2(i)] where T is shared table, h is hash, s is random sign. Adam optimizes K entries instead of N. Two-hash scheme with sign flips provides degrees of freedom despite collisions.
- **Convergence speedup:** 2-4x (reduced noise, effective LR per ternary bit is R times higher). Memory: 4-16x reduction.
- **Feasibility:** HIGH — drop-in replacement, ~50 lines code change in TernaryLinear. Test R = 4, 8, 16, 32.
- **Synergies:** Compatible with ALL STE variants. Combinable with LWFP (P5, per-layer hash tables) and CFS (P3, flip scheduling on hashed values).

### R2-A3-P3: CFS — Combinatorial Flip Scheduling
- **Paradigm shift:** Stop continuous optimization. Directly compute loss-reduction from each possible ternary flip and select the best set per step. NO shadow weights, NO Adam, NO STE.
- **Key insight:** Each ternary weight has at most 2 valid transitions. Gradient provides first-order loss change estimate. Rank all transitions by |delta_L|, select top-B (flip budget), apply directly. Information flow rate = B * 1.585 bits/step — guaranteed and controllable.
- **Second-order enhancement:** For top-2B candidates, compute randomized Hessian-vector product to estimate interaction effects. Reject candidates with negative interaction.
- **Flip cooldown:** Weight that just flipped cannot flip again for C steps (discrete analogue of momentum, prevents oscillation).
- **Convergence speedup:** 2-5x. Memory: 6-12x (no shadow weights, no optimizer state).
- **Feasibility:** HIGH — ~100 lines PyTorch. Replace optimizer loop with flip-selection loop.
- **Synergies:** Natural partner for LWFP (P5, per-layer flip budgets) and EDTT (P4, timing + selection).

## TIER 2 — MEDIUM-HIGH VIABILITY (test after Tier 1)

### R2-A3-P4: EDTT — Event-Driven Threshold Training
- **Paradigm shift:** Replace uniform-time gradient descent with discrete event simulation. Only compute when a weight ACTUALLY crosses a threshold. Skip all informationally empty steps.
- **Key insight:** Between threshold crossings, the ternary loss is CONSTANT — gradient steps that don't flip any weight produce zero change in model output. At any step, only 1-5% of weights are near a threshold. 95-99% of gradient computation is wasted.
- **Method:** For each weight, compute time_to_flip = distance_to_threshold / |EMA_gradient|. Maintain min-heap ordered by time_to_flip. Pop events, apply flips, recompute affected gradients. Full recalibration every K events.
- **Adaptive batch:** When few weights near threshold → increase LR to push more toward threshold. When many near simultaneously → slow down to avoid avalanche.
- **Convergence speedup:** 5-20x in later training. 1-2x in early training (when everything is flipping).
- **Feasibility:** MEDIUM-HIGH — mathematical framework straightforward, main challenge is efficient sparse backward passes.
- **Synergies:** ACSI (P1) provides near-optimal init → fewer thresholds to cross → EDTT even more efficient in polishing.

### R2-A3-P5: LWFP — Layer-Wise Fictitious Play
- **Paradigm shift:** Treat each LAYER as an independent game-theoretic agent. Each layer optimizes its ternary weights assuming others are fixed, with volatility-dependent flip budgets. Convergence order EMERGES from game dynamics.
- **Key insight:** Neural network training is a potential game (shared loss function). Fictitious play in potential games converges to Nash equilibrium. Volatility-based flip budget: B_L = B_base * exp(-alpha * V_L) where V_L = flip count in adjacent layers.
- **Crystallization fronts:** Stable layers broadcast stability signal → adjacent layers increase flip budget → convergence propagates layer-by-layer. Simple layers (embedding, projection) converge first, complex layers (middle) last.
- **Convergence speedup:** 2-4x. Focuses compute on layers that are actively changing.
- **Feasibility:** MEDIUM-HIGH — modest modification of training loop. Add per-layer flip tracking and volatility-dependent budgets.
- **Synergies:** Layers on top of CFS (P3, per-layer flip scheduling) or HSC (P2, per-layer hash tables).

### R2-A3-P6: TRA — Ternary Reservoir Architecture
- **Paradigm shift:** Do NOT train most of the network. Use random ternary matrices for hidden layers (provably good random projections via JL lemma and RIP). Only train output/readout layers.
- **Key insight:** Random ternary matrix W in {-1,0,+1}^{m x n} with uniform distribution approximately preserves norms and inner products (Marchenko-Pastur law). Stacking random projections with nonlinearities creates random feature space that already separates data.
- **For SSM:** Random ternary A creates random linear dynamical system with bounded spectral radius. Stacking multiple random SSM layers = bank of random temporal filters covering broad timescales.
- **Quality recovery:** Progressive unfreezing (output to input), depth scaling (48 random layers ~ 24 trained), feature selection (LASSO on fixed features).
- **Convergence speedup:** 10-50x (5-12x fewer trainable params + near-convex readout optimization).
- **Feasibility:** HIGH for prototyping (trivially testable — just freeze layers). MEDIUM for production quality (may need 3-5x depth increase).
- **Synergies:** ACSI (P1) for readout construction → potentially zero training.

### R2-A3-P7: SQRT — Symmetry-Projected Quotient Training
- **Paradigm shift:** The ternary weight space has MASSIVE symmetry group G (neuron permutations S_n, sign flips Z_2^n, zero equivalences). Standard training wastes steps along symmetry orbits. SQRT projects gradients onto the quotient space.
- **Key insight:** For n=256 neurons, |G| = n! * 2^n ~ 10^535. Two weight configurations related by symmetry compute the SAME function. Canonicalization: sort neurons by weight signature + fix sign convention → every gradient step is guaranteed to change the FUNCTION.
- **Projected gradient:** g_quotient = g - P_G(g). For neurons with same fingerprint, average their gradients (ensures equivalent neurons evolve together).
- **Convergence speedup:** 2-5x. Higher in early training (many equivalent neurons), lower in late training.
- **Feasibility:** MEDIUM-HIGH — approximate canonicalization (sort by norm/fingerprint) captures 90%+ benefit.
- **Synergies:** Orthogonal to all other paradigms (reduces search space, benefits any optimizer). Powerful with CCD (P8, cleaner phase transitions in quotient space).

## TIER 3 — MEDIUM VIABILITY (speculative)

### R2-A3-P8: CCD — Critical Cluster Dynamics
- **Paradigm shift:** Treat ternary weights as a 3-state Potts spin glass. Replace single-weight-flip (Glauert dynamics) with correlated CLUSTER flips (Swendsen-Wang dynamics). Achieves critical speedup near phase transitions.
- **Key insight:** Swendsen-Wang achieves z ~ 0 exponent at criticality vs z > 2 for single-flip methods. Build weight interaction matrix J_ij from Hessian, create bonds with p_ij = 1 - exp(-beta * J_ij), cluster via union-find, evaluate 3 possible assignments per cluster.
- **Critical temperature:** Maintain beta near beta_c where cluster size distribution follows power law P(s) ~ s^{-2.18} (3D Potts universality).
- **Convergence speedup:** 2-15x per step. Each step 3-10x more expensive. Net: 2-5x wall-clock.
- **Feasibility:** MEDIUM — physics well-established for Potts model, less certain for NN loss landscapes.
- **Synergies:** EDTT (P4, cluster events) + SQRT (P7, cleaner quotient-space phase transitions).

### R2-A3-P9: RDC — Reaction-Diffusion Crystallization
- **Paradigm shift:** Replace gradient descent with PDE dynamics. Triple-well potential V(s) = (s^2 - 1)^2 * s^2 / 4 has stable fixed points at {-1,0,+1}. Gradient signal acts as external field. Spatial diffusion creates coherent ternary domains.
- **Key insight:** ds/dt = -s(s^2-1)(2s^2-1) + D*Laplacian(s) - eta*grad. Weights SNAP to ternary values as reaction strength increases — no explicit quantization needed. Quantization is EMERGENT. Domain walls move as coherent fronts, reassigning thousands of weights per step.
- **No STE needed:** Dynamics are smooth and differentiable everywhere. The gradient through quantization problem disappears.
- **Convergence speedup:** 2-5x if weight matrices have spatial structure. 1-2x even without (smooth triple-well > hard threshold + STE).
- **Feasibility:** MEDIUM — elegant math (Allen-Cahn equation), but depends on spatial coherence in weight matrices.
- **Synergies:** Reaction term alone (no diffusion) is a drop-in STE replacement.

## UNIFYING INSIGHT

**The Informationally Empty Step Problem:** In standard ternary training, the flip rate (fraction of weights crossing a threshold per step) is typically 1-5%. This means 95-99% of gradient computation produces ZERO change in model output. Every paradigm above attacks this waste:

| Approach | Strategy | Speedup |
|----------|----------|---------|
| Standard (STE+Adam) | Compute everything every step | 1x |
| EDTT (P4) | Skip non-events | 5-20x |
| CFS (P3) | Directly select best flips | 2-5x |
| HSC (P2) | Compress optimizer space | 2-4x |
| LWFP (P5) | Focus on active layers | 2-4x |
| TRA (P6) | Don't train most layers | 10-50x |
| ACSI (P1) | Construct analytically, polish only | 20-100x |

## KEY SYNERGIES AND RECOMMENDED COMBINATIONS

### Conservative Path (HIGH viability)
HSC (P2) + CFS (P3): Hashed shadow weights for noise reduction + direct flip scheduling. Drop-in replacements, ~150 lines total code. Expected: 3-8x speedup.

### Aggressive Path (MEDIUM-HIGH viability)
ACSI (P1) + EDTT (P4): Analytical construction of SSM weights + event-driven polishing. Construction provides near-optimal init (seconds), EDTT efficiently fixes remaining misalignments. Expected: 50-200x speedup.

### Moonshot Path
TRA (P6) + ACSI (P1): Random hidden layers + analytically constructed readout. Potentially ZERO gradient training needed. Expected: 100x+ if quality holds.

## ASPECT 3 COMPLETE — SUMMARY ACROSS ALL WAVES

### Wave 1 (Web): 12 findings on optimizers, schedules, initialization, stability techniques
### Wave 2 (Math): 12 novel formulations (optimal LR, STE bias correction, adaptive freezing, EOTS, mirror descent, Hessian, spectral, CGA, Langevin, phase coupling, belief propagation, TLMC)
### Wave 3 (Paradigm): 9 radical alternatives (ACSI, HSC, CFS, EDTT, LWFP, TRA, SQRT, CCD, RDC)

**Total unique findings Aspect 3: 33**

### TOP 5 Most Impactful for Our Project:
1. **ACSI (P1)** — analytical SSM weight construction via Ho-Kalman, 20-100x, exploits linearity of SSM
2. **HSC (P2)** — hashed shadow weight compression, 2-4x, 50 lines, drop-in replacement
3. **CFS (P3)** — direct flip scheduling without shadow weights, 2-5x, eliminates Adam/STE
4. **EDTT (P4)** — event-driven training skipping empty steps, 5-20x in later training
5. **TRA (P6)** — random ternary reservoir with trained readout, 10-50x if quality holds
