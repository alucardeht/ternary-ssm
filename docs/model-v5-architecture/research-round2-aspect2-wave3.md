# Research Round 2, Aspect 2: Training Memory — Wave 3 (Paradigm-Breakers)

## Summary

Two paradigm-breakers explored radical alternatives to conventional training memory models. After deduplication, 10 unique paradigm proposals remain. The unifying theme: the 85 GB memory budget exists because we treat ternary weights as continuous — every GB of shadow weights, momentum, and variance tracks continuous dynamics in a fundamentally DISCRETE space. Algorithms native to the ternary lattice collapse memory by orders of magnitude.

## TIER 1 — HIGH VIABILITY (test first)

### R2-A2-P1: Branch Predictor Training (BPT) — 2-Bit Saturating Counters
- **Paradigm shift:** Replace Adam (128 bits/param) with a 2-bit saturating counter per weight, inspired by CPU branch prediction hardware. Counter states {00,01,10,11} map to ternary values {-1,0,0,+1}. Hysteresis prevents oscillation.
- **Key insight:** A 2-bit counter implements a nonlinear low-pass filter on gradient signs. P(+1) = q^2/(q^2+(1-q)^2) where q = P(positive gradient). When q > 0.71, weight settles at +1. When 0.29 < q < 0.71, weight stays at 0. Natural threshold without tuning.
- **Memory at 7B:** 2-bit counters (1.75 GB) + packed ternary (1.40 GB) + current layer gradient signs (12 MB) + activations (0.5 GB) = **~3.75 GB total** (22.7x reduction)
- **Feasibility:** MEDIUM-HIGH — trivially implementable (2 bitwise ops per weight per step). Risk is convergence speed — may need 10-50x more steps.
- **Synergies:** Combines with RTN (P4) → <2 GB. Extreme version of TMC counters (R2-A1-M2).

### R2-A2-P2: RTRL for Ternary SSM — Forward-Only Exact Gradients
- **Paradigm shift:** Compute gradients FORWARD in time alongside the hidden state, eliminating backward pass and activation storage. Standard RTRL is O(N^4) and impractical. For diagonal SSM + ternary weights: O(N * f * P_layer) where f ≈ 1% (near-threshold weights).
- **Key insight:** Three properties collapse RTRL complexity: (1) Diagonal A → O(N) sensitivity update, (2) ternary STE sparsity → only ~1% of weights have non-zero derivative, (3) layer pipelining → one layer's sensitivities at a time.
- **Memory at 7B:** Sensitivity states (168 MB) + current layer weights/optimizer (256 MB) = **~365 MB GPU active**. Shadow weights and full model on SSD.
- **Compute:** 256 * 160K * 100 layers * 512 tokens = 2.1 TFLOPs/sequence → ~17ms on RTX 5070. Comparable to BPTT.
- **Feasibility:** MEDIUM-HIGH — mathematically sound, exploits UNIQUE properties of diagonal SSM + ternary. Main risk: sparsity fraction f may be higher early in training.
- **Synergies:** TSCS (ternary A) → A * S becomes add/sub. TPGC (M8) identifies active weight set. ABPO (M6) provides pipelining.

### R2-A2-P3: Regenerative Weight Codec (RWC)
- **Paradigm shift:** Never store 7B weights as an array. Instead, maintain a compact generating function (codec) that produces ternary weights on demand given position indices. Train the codec parameters (52M), not the weights (7B). Weights are ephemeral — generated per forward pass, discarded immediately.
- **Key insight:** W_l = Q(U_l @ V_l^T) where Q = ternary quantization. Rank-64 logit factorization can generate full-rank ternary matrices (quantization breaks rank structure). 52M codec params → 7B ternary weights = 134x compression.
- **Memory at 7B:** Codec FP16 (105 MB) + Adam for codec (629 MB) + current layer generated weights (4 MB) = **~740 MB total**. Under 1 GB.
- **Feasibility:** MEDIUM-HIGH — well-understood low-rank factorization + quantization. Risk: effective capacity may be codec size (52M) not architecture size (7B).
- **Synergies:** ABPO (M6) for layer pipelining, INSTANT (F6) for codec gradient compression.

### R2-A2-P4: Reversible Ternary Network (RTN) — Zero Activation Storage
- **Paradigm shift:** Make the entire forward pass invertible by exploiting ternary add/sub exactness. RevNet-style coupling: x1_out = x1 + F(x2), x2_out = x2 + G(x1_out). Inversion is EXACT for ternary (integer add/sub has no floating-point error). Activations recovered by running backward from output.
- **Key insight:** For continuous weights, RevNet has numerical error accumulation. For TERNARY, the forward computation is exact integer arithmetic → inversion is also exact. This is a property UNIQUE to ternary/binary networks.
- **Memory impact:** Eliminates 4-20 GB activation storage. As standalone: modest. Combined with BPT (P1): 3.27 GB total. Combined with LSCD (P5): 1.94 GB total.
- **Feasibility:** HIGH (math), MEDIUM (architecture) — requires RevNet coupling structure that halves effective width. Mitigation: increase d_model by sqrt(2).
- **Synergies:** CRITICAL enabler for P1 (BPT) and P5 (LSCD). Without RTN, those still need activation memory.

## TIER 2 — MEDIUM VIABILITY (test after Tier 1)

### R2-A2-P5: Layer-Sequential Coordinate Descent (LSCD)
- **Paradigm shift:** Train one layer at a time, all others frozen as packed ternary (0.2 bytes/param). Peak memory = LARGEST SINGLE LAYER, not total model.
- **Key insight:** For SSM layers, single-layer optimization is a linear system identification problem with closed-form solution via least-squares. Solve continuous problem → quantize to ternary → iterate on residual.
- **Memory at 7B:** Frozen model (1.40 GB) + active layer shadow+Adam (536 MB) + activations (1 GB) = **~2.94 GB** (28.9x reduction)
- **Feasibility:** MEDIUM — blockwise training works (Hinton 2006 deep belief nets) but quality gap vs end-to-end is unknown for LLMs.
- **Synergies:** BPT (P1) for per-layer optimizer → 1.48 GB. RTN (P4) eliminates activation memory → 1.94 GB.

### R2-A2-P6: Causal Gradient Streaming (CGS)
- **Paradigm shift:** Compute gradient contributions DURING the forward pass, streaming through the network. Use token t's activations as proxy for token t-1's activations in backward computation. Error is O(spectral_radius(A)) — small for stable SSMs.
- **Key insight:** SSM hidden state h_t ≈ h_{t-1} (one token's difference). Proxy gradient has bounded bias that decays exponentially with SSM decay rate.
- **Memory at 7B:** Current token activations (1.6 MB) + gradient signals (1.6 MB) + current layer optimizer (192 MB) = **~260 MB GPU active**
- **Feasibility:** MEDIUM — sound for fast-decaying SSM states. Risk: SSMs needing long-range memory (spectral radius near 1) have larger proxy bias.
- **Synergies:** Refines ABPO (M6) with cheaper activation handling. Complements RTRL (P2) as a simpler approximation.

### R2-A2-P7: Low-Rank Logit Parametrization (LRLP)
- **Paradigm shift:** Decompose ternary weight decisions into sign + mask, each parameterized by low-rank logit matrices. Z_S = U_S V_S^T for signs, Z_M = U_M V_M^T for sparsity mask. W = round(sigmoid(Z_S)) * round(sigmoid(Z_M)).
- **Key insight:** Rank of the LOGIT matrix ≠ rank of the generated TERNARY matrix. Round(sigmoid(.)) is highly nonlinear → rank-64 logit → any-rank ternary. Low-rank logit does NOT limit ternary expressivity.
- **Memory at 7B:** LRLP factors + Adam: 7.56 GB (ternary part) + 13.4 GB (FP16 Adam) + 1.4 GB (packed ternary) + 2 GB (activations) = **~24.4 GB** (3.5x reduction)
- **Feasibility:** HIGH — straightforward PyTorch implementation, rank R is tunable quality-memory knob.
- **Synergies:** GaLore for factor gradient compression. Similar to RWC (P3) but with explicit sign/mask decomposition.

## TIER 3 — SPECULATIVE (moonshot)

### R2-A2-P8: Gibbs-Boltzmann Ternary Training (GBTT)
- **Paradigm shift:** Replace gradient-based optimization with statistical mechanics. Each weight is a 3-state particle. Training = cooling the system. Block Gibbs sampling: update entire row simultaneously with 3 forward passes (exploiting row independence given input).
- **Memory at 7B:** Current layer weights (4 MB) + activations (4 MB) = **~8 MB GPU active** (with full model on SSD)
- **Bottleneck:** SSD loading — 3 passes × 4096 rows × 100 layers × 0.25s = 3.5 DAYS per sweep. Mitigated by importance sampling.
- **Feasibility:** LOW-MEDIUM — extraordinary memory (<50 MB GPU for 7B) but convergence speed unknown. Best as fine-tuning method.

### R2-A2-P9: Stochastic Ternary Automata (STA) — Local Hebbian Rules
- **Paradigm shift:** Abandon gradients entirely. Each weight updates based on LOCAL information only (pre/post-synaptic activity). Rule: flip probability = sigma(a_k * x_j * (y_target - y_i)). Only 3 global learnable scalars for the entire update rule.
- **Memory at 7B:** Packed ternary (1.40 GB) + current layer activations (128 MB) = **~1.53 GB** (55.6x reduction)
- **Feasibility:** LOW-MEDIUM — moonshot. Forward-forward algorithm achieves ~1-2% lower accuracy on simple tasks. Gap at 7B LLM scale unknown. Ternary discreteness makes this MORE viable than for continuous weights.

### R2-A2-P10: Ternary Weight Tournament (TWT) — Evolutionary Population
- **Paradigm shift:** Maintain population of P=50 models on SSD (87.5 GB SSD). No gradients, no backward pass. Tournament selection + structured crossover (row-level) + activation-magnitude-guided mutation.
- **Memory at 7B:** ONE model in GPU (1.75 GB) + activations (100 MB) = **~1.85 GB GPU**. Population on SSD (87.5 GB).
- **Feasibility:** LOW-MEDIUM — evolutionary strategies are 10-100x less sample-efficient than gradient methods. Viable as polishing step after gradient pre-training.

## UNIFYING INSIGHT

**The Ternary Memory Paradox:** Current training spends 85 GB to navigate a space that has only log2(3^{7B}) = 11.1 GB of possible configurations. The 7.7x overhead comes from using continuous-space tools (FP32 shadow weights, Adam momentum/variance) to solve a discrete problem.

**The compression opportunities correlate with ternary-specificity:**

| Approach | GPU Memory (7B) | Reduction | Uses Gradients? |
|----------|-----------------|-----------|-----------------|
| Standard (Adam + BPTT) | 85 GB | 1x | Yes (FP32) |
| LRLP (P7) | 24.4 GB | 3.5x | Yes (low-rank) |
| BPT (P1) | 3.75 GB | 22.7x | Sign only |
| LSCD (P5) | 2.94 GB | 28.9x | Yes (1 layer) |
| TWT (P10) | 1.85 GB | 46x | No |
| STA (P9) | 1.53 GB | 55.6x | No (local) |
| RWC (P3) | 740 MB | 115x | Yes (codec) |
| RTRL-TSSM (P2) | 365 MB | 233x | Yes (forward) |
| CGS (P6) | 260 MB | 327x | Approx (proxy) |
| GBTT (P8) | 8 MB | 10,625x | No (sampling) |

## KEY SYNERGIES AND RECOMMENDED COMBINATIONS

### Conservative Path (HIGH viability)
RWC (P3) + RTN (P4): ~600 MB for 7B. Codec generates weights on demand + reversible computation eliminates activations. Both techniques independently testable.

### Aggressive Path (MEDIUM viability)
BPT (P1) + RTN (P4) + LSCD (P5): ~1.9 GB for 7B. Branch predictor training + reversible ternary + layer-sequential. Any two of three combinable independently.

### Moonshot Path
RTRL-TSSM (P2) + RWC (P3): ~400 MB for 7B. Forward-only exact gradients for SSM layers + codec parametrization. Exploits the deepest mathematical properties of the architecture.

## ASPECT 2 COMPLETE — SUMMARY ACROSS ALL WAVES

### Wave 1 (Web): 9 findings on optimizer compression, activation quantization, offloading, encoding
### Wave 2 (Math): 10 novel formulations (SPRT accumulator, spectral sharding, noise-floor allocation, fused cascade, interval scheduling, ABPO, ternary checkpointing, TPGC, packed state trajectory, holographic memory)
### Wave 3 (Paradigm): 10 radical alternatives (BPT, RTRL-TSSM, RWC, RTN, LSCD, CGS, LRLP, GBTT, STA, TWT)

**Total unique findings Aspect 2: 29**

### TOP 5 Most Impactful for Our Project:
1. **RTRL-TSSM (P2)** — forward-only exact gradients, 365 MB, exploits diagonal SSM + ternary sparsity
2. **RWC (P3)** — codec parametrization, 740 MB, 134x weight compression via low-rank generation
3. **ABPO (M6)** — 7B in ~3 GB GPU via bandwidth asymmetry exploitation
4. **BPT (P1)** — 2-bit counters replacing Adam, 3.75 GB, ternary-native optimizer
5. **Fused Gradient Cascade (M4)** — zero gradient buffer, 5.3x total system reduction
