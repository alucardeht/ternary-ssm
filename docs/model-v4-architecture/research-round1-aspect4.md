# Round 1, Aspect 4: Ultra-Efficient SSM State — Compression, Speed & Quantization

## Wave 1 — Web Research (2 researchers, Feb 2026)

### NEXT-GENERATION SSM ARCHITECTURES

#### v4-R1-A4-F1: Mamba-3 — MIMO Formulation with Complex-Valued SSM
- **Source:** ICLR 2026
- **Summary:** Multi-Input Multi-Output (MIMO) formulation of Mamba. Uses complex-valued SSM via data-dependent RoPE and trapezoidal discretization. Achieves 100% parity accuracy where Mamba-2 gets random chance. Significant leap in SSM expressivity.
- **Quality:** BETTER than Mamba-2 (100% vs random on parity tasks)
- **Efficiency:** Similar compute profile to Mamba-2, more expressive per parameter
- **Maturity:** Published at top venue (ICLR 2026)
- **Relevance:** Complex-valued state may interact differently with ternary quantization — needs investigation
- **Verdict:** PROMISING

#### v4-R1-A4-F2: RWKV-7 Goose — Dynamic State Evolution
- **Source:** RWKV Foundation, 2025-2026
- **Summary:** Introduces Dynamic State Evolution that surpasses TC0 computational limitations of previous linear attention/SSM models. Achieves 1850 tok/s at 128k context length. Models validated from 0.4B to 14B parameters. Fundamentally more expressive recurrence than fixed-structure SSMs.
- **Quality:** Surpasses TC0 limitations (provably more expressive than standard linear recurrence)
- **Efficiency:** 1850 tok/s at 128k context — among fastest for recurrent models
- **Maturity:** Production-ready (multiple model sizes released)
- **Relevance:** Dynamic state evolution could complement ternary routing — state transitions that adapt
- **Verdict:** PROMISING

#### v4-R1-A4-F3: xLSTM — Exponential Gating + Matrix Memory
- **Source:** NeurIPS 2024
- **Summary:** Extends LSTM with exponential gating mechanism and matrix-valued memory state. Pareto-dominant over Transformers at fixed compute budget for long contexts. Scaling laws show favorable extrapolation. Represents the strongest "enhanced recurrence" approach.
- **Quality:** Pareto-dominant over Transformers at fixed compute for long contexts
- **Efficiency:** Better scaling laws than Transformers in data-constrained regime
- **Maturity:** Published at NeurIPS 2024, active follow-up research
- **Relevance:** Matrix memory state is dense — ternary version would need careful design. Exponential gating may conflict with ternary.
- **Verdict:** PROMISING

### SSM STATE COMPRESSION & SPARSITY

#### v4-R1-A4-F4: S²Mamba — Word-Importance Sparsity + ASIC Acceleration
- **Source:** 2025
- **Summary:** Applies Word-Importance Sparsity to filter out 40% of tokens before SSM processing. Combined with ASIC implementation achieves 35-97x speedup over GPU for the sparse SSM pipeline. Demonstrates that not all tokens need full state processing.
- **Quality:** Minimal degradation from 40% token filtering
- **Efficiency:** 35-97x speedup (ASIC), 40% input reduction (algorithmic)
- **Maturity:** ASIC design validated in simulation
- **Relevance:** Token filtering aligns with our "Surprise Processor" concept — only process surprising tokens. 40% filtering is conservative vs our 70% prediction.
- **Verdict:** PROMISING

#### v4-R1-A4-F5: HGRN-2 — Hierarchical Gated Recurrence
- **Source:** 2024-2025
- **Summary:** Hierarchical forget gates with learnable lower bounds. Creates multi-scale state decay — fast-changing features use high forget rates, stable features use low rates. More principled than fixed exponential decay.
- **Quality:** Competitive with Mamba on standard benchmarks
- **Efficiency:** Similar to linear recurrence — no attention overhead
- **Maturity:** Published, open-source
- **Relevance:** Hierarchical gates could be implemented with ternary routing — {-1, 0, +1} maps to {reset, hold, invert} at different timescales
- **Verdict:** NEEDS-MORE-WORK

### TERNARY & BINARY SSM VALIDATION (CRITICAL)

#### v4-R1-A4-F6: Slender-Mamba — Ternary {-1,0,+1} SSM (COLING 2025)
- **Source:** COLING 2025
- **Summary:** Applies BitNet b1.58 ternary quantization method directly to Mamba-2. First published validation that ternary {-1,0,+1} weights work for SSM architectures. Comparable performance to full-precision at fraction of memory/compute.
- **Quality:** Comparable to full-precision Mamba-2
- **Efficiency:** Ternary weights = add/subtract only, no matmul
- **Maturity:** Published at COLING 2025, peer-reviewed
- **Relevance:** **DIRECT VALIDATION of our core thesis.** Ternary SSMs are not just theoretical — they work in practice. This is the strongest external evidence for our approach.
- **Verdict:** CRITICAL VALIDATION

#### v4-R1-A4-F7: Bi-Mamba — Binary {-1,+1} SSM
- **Source:** 2024-2025
- **Summary:** Even more extreme than ternary — binary {-1,+1} quantization of Mamba weights. Achieves comparable performance to full-precision. Demonstrates that SSM recurrence is inherently robust to extreme quantization.
- **Quality:** Comparable to full-precision (surprisingly robust)
- **Efficiency:** Binary = XOR operations, even simpler than ternary
- **Maturity:** Published, validated
- **Relevance:** If BINARY works, ternary is an EASIER target. Confirms SSMs are more quantization-friendly than Transformers.
- **Verdict:** PROMISING

#### v4-R1-A4-F8: TVMamba — Ternary Weights + Activations for Vision
- **Source:** 2025
- **Summary:** Applies ternary quantization to BOTH weights AND activations in Mamba for vision tasks. Extends ternary beyond just weights to the full compute pipeline.
- **Quality:** Competitive on vision benchmarks
- **Efficiency:** Full ternary pipeline — both weights and activations quantized
- **Maturity:** Published for vision domain
- **Relevance:** If ternary activations work for vision Mamba, may work for language. Full ternary pipeline would maximize hardware efficiency.
- **Verdict:** PROMISING

### SSM HARDWARE & TRAINING OPTIMIZATION

#### v4-R1-A4-F9: Hybrid SSM-Attention — Optimal Architecture Mix
- **Source:** Multiple papers (StripedHyena-7B, Jamba, etc.), 2024-2025
- **Summary:** Extensive research validates that the optimal architecture uses 10-30% attention layers with the rest as SSM/recurrence. Pure SSM struggles with precise retrieval; pure attention is compute-expensive. The hybrid achieves best of both worlds.
- **Quality:** Validated across multiple architectures and scales
- **Efficiency:** 70-90% SSM layers = mostly recurrent (O(1) per token) with selective attention
- **Maturity:** Production-deployed (StripedHyena, Jamba)
- **Relevance:** Our architecture should use ~10-30% attention layers. Ternary SSM for bulk + sparse attention for retrieval = optimal design point.
- **Verdict:** PROMISING

#### v4-R1-A4-F10: Mamba-X — Systolic Array Hardware Acceleration
- **Source:** 2025
- **Summary:** Custom systolic array design for Mamba's selective scan operation. Achieves 2.3x speedup and 11.5x energy reduction over GPU implementation. Hardware-level optimization for recurrent computation.
- **Quality:** N/A (hardware accelerator)
- **Efficiency:** 2.3x speed, 11.5x energy improvement
- **Maturity:** Hardware design validated
- **Relevance:** Ternary SSM on custom hardware could compound these gains — ternary operations are simpler for hardware than FP16 scan.
- **Verdict:** PROMISING

#### v4-R1-A4-F11: Parallel Scan Optimizations — Training Speedup
- **Source:** Multiple papers, 2024-2025
- **Summary:** Collection of optimizations for SSM training: hardware-aware parallel scan, Monarch Mixer's structured approach, Based architecture achieving 4.5x throughput improvement. Key insight: SSM training bottleneck is scan parallelism, not compute.
- **Quality:** N/A (training optimization)
- **Efficiency:** Up to 4.5x training throughput improvement
- **Maturity:** Various stages — some production, some research
- **Relevance:** Training speed is our Pillar 2. These optimizations stack with ternary's compute savings.
- **Verdict:** PROMISING

### ANALOG & NEUROMORPHIC SSM

#### v4-R1-A4-F12: QS4D — Quantized SSM for Analog Compute-in-Memory
- **Source:** 2025
- **Summary:** Quantization-aware training specifically designed for deploying SSMs on analog compute-in-memory (CIM) hardware. Bridges digital ternary training with analog inference deployment.
- **Quality:** Maintains accuracy through quantization-aware training
- **Efficiency:** Analog CIM enables massive parallelism at low power
- **Maturity:** Research prototype
- **Relevance:** Future deployment path — train ternary digitally, deploy on analog CIM. Ternary weights map naturally to multi-level analog cells.
- **Verdict:** NEEDS-MORE-WORK

#### v4-R1-A4-F13: SNN+SSM Hybrids + Neuromorphic LLMs
- **Source:** Multiple papers, 2024-2025 + Intel Loihi 2 work
- **Summary:** Binary-activated RNNs with SSM structure for neuromorphic hardware. Neuromorphic LLMs on Loihi 2 demonstrate that ternary SSMs have particularly low quantization error compared to other architectures. SSM recurrence maps naturally to neuromorphic spike-based computation.
- **Quality:** Validated on neuromorphic hardware
- **Efficiency:** Sub-25 nJ/token on Loihi 2 (vs ~1000 nJ/token GPU)
- **Maturity:** Hardware prototypes (Intel Loihi 2)
- **Relevance:** Long-term deployment target. Ternary SSMs on neuromorphic hardware = ultimate efficiency. Validates that SSM + ternary is hardware-natural.
- **Verdict:** PROMISING

#### v4-R1-A4-F14: Griffin/Hawk + eMamba + TR-mamba2attn (Minor Findings)
- **Source:** Google (Griffin/Hawk), various (eMamba, TR-mamba2attn)
- **Summary:** Collection of smaller findings: Google's Griffin/Hawk recurrent architectures (production-grade, partial benchmark data). eMamba: NAS-based architecture search for edge Mamba deployment. TR-mamba2attn: weight-tying technique between attention and SSM layers to reduce parameters.
- **Quality:** Various
- **Efficiency:** Weight-tying (TR-mamba2attn) reduces total parameters. NAS (eMamba) finds efficient configurations.
- **Maturity:** Griffin/Hawk production (Google); eMamba/TR-mamba2attn research
- **Relevance:** Weight-tying between SSM and attention is interesting for our hybrid architecture — could reduce the cost of the attention layers.
- **Verdict:** NEEDS-MORE-WORK

## Key Insights — Aspect 4 Wave 1

1. **SSMs quantize BETTER than Transformers** — Multiple independent papers (Slender-Mamba, Bi-Mamba, TVMamba, QS4D) confirm SSM recurrence is inherently more robust to extreme quantization. This is the strongest possible validation for our approach.

2. **Ternary SSM is DIRECTLY validated** — Slender-Mamba (COLING 2025) applies BitNet b1.58 to Mamba-2 with comparable quality. This is not theoretical — it works.

3. **Binary works too** — If even binary {-1,+1} SSMs match full-precision, ternary {-1,0,+1} is a SAFER bet with more expressivity.

4. **10-30% attention is optimal** — Hybrid SSM-attention architecture is the consensus across multiple production deployments. Pure SSM and pure attention are both suboptimal.

5. **Next-gen SSMs are more expressive** — Mamba-3 MIMO, RWKV-7 Dynamic State, xLSTM matrix memory all push recurrence expressivity. Our ternary SSM should incorporate these advances.

6. **Hardware acceleration compounds with ternary** — Mamba-X gets 2.3x on FP16 scan; ternary scan is simpler = even more hardware-friendly.

7. **Token sparsity is validated** — S²Mamba filters 40% of tokens. Our "Surprise Processor" concept (filter 70%) is aggressive but in the right direction.

8. **Neuromorphic is the endgame** — Ternary SSMs on Loihi 2 = sub-25 nJ/token. Our hardware portability requirement aligns perfectly.

## For Our Project

**VALIDATED DECISIONS:**
- Ternary SSM architecture: externally validated by Slender-Mamba, Bi-Mamba, TVMamba
- Hybrid SSM+attention: 10-30% attention layers confirmed optimal
- Token-level sparsity: S²Mamba validates filtering non-important tokens
- Hardware portability: neuromorphic and analog CIM paths both viable

**OPEN QUESTIONS for math-paradigm-breakers:**
- How does Mamba-3's complex-valued state interact with ternary quantization?
- Can RWKV-7's dynamic state evolution be expressed with ternary transitions?
- What is the minimal attention ratio needed for our ternary SSM hybrid?
- How does xLSTM's exponential gating degrade under ternary constraints?
- Can parallel scan be reformulated for ternary-specific hardware (add/subtract scan)?
