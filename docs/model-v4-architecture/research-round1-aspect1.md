# Round 1, Aspect 1: Non-Autoregressive & Alternative Generation Paradigms

## Wave 1 — Web Research (2 researchers, Feb 2026)

### TRUE PARADIGM SHIFTS (quality matches AR)

#### v4-R1-A1-F1: LLaDA — First Diffusion LM Matching AR Quality
- **Source:** arXiv 2502.09992v1, February 2025
- **Summary:** 8B parameter masked diffusion model trained on 2.3T tokens. FIRST to match LLaMA3 8B on MMLU (65.9 vs 65.4), surpasses on GSM8K (70.7 vs 53.1). Breaks reversal curse — outperforms GPT-4o on reversal tasks. Uses masked diffusion (not autoregressive).
- **Quality:** SAME as LLaMA3 8B (matches or exceeds on most benchmarks)
- **Efficiency:** Training cost equivalent to AR. Inference uses parallel token prediction.
- **Maturity:** Production-ready (deployed with API)
- **Verdict:** PROMISING

#### v4-R1-A1-F2: Mercury 2 — 5x Faster Diffusion LM (Production)
- **Source:** Inception Labs, February 25, 2026
- **Summary:** Production diffusion LM: 1,009 tok/s on Blackwell, 5x faster than Claude Haiku 4.5 / GPT-5 mini (~200 tok/s). 128K context. Competitive quality on benchmarks.
- **Quality:** CLOSE to frontier speed-optimized models (GPT-5 mini, Claude Haiku 4.5)
- **Efficiency:** 5x faster inference, lower price ($0.25/M input)
- **Maturity:** Production-ready (early access Feb 2026)
- **Verdict:** PROMISING (proves diffusion CAN be faster than AR at production scale)

#### v4-R1-A1-F3: Discrete Diffusion Scaling Laws
- **Source:** ETH Zurich, arXiv 2512.10858v3, December 2024
- **Summary:** First comprehensive scaling study (25M-10B params). Uniform diffusion requires MORE parameters but LESS data than AR for same compute. Favorable in data-constrained settings. Parity validated up to 10B.
- **Quality:** SAME at scale (extrapolations show parity at 10B+)
- **Efficiency:** More parameter-heavy scaling = better token efficiency (important as datasets saturate)
- **Maturity:** Proof-of-concept (code open-sourced)
- **Verdict:** PROMISING

#### v4-R1-A1-F4: Jacobi Forcing — Parallel Causal Decoding
- **Source:** arXiv 2512.14681, December 2025
- **Summary:** Uses Jacobi iteration from numerical analysis for parallel token generation while maintaining causal structure. Multiple tokens predicted then iteratively refined. Works with standard AR-trained models (no retraining).
- **Quality:** CLOSE (claims AR-level with parallel generation)
- **Efficiency:** 2-3x faster than standard AR
- **Maturity:** Production-ready (GitHub repo, works with pretrained models)
- **Verdict:** PROMISING

#### v4-R1-A1-F5: One-Step Continuous Denoising
- **Source:** arXiv 2602.16813v1, February 2026
- **Summary:** Novel continuous denoising for single-step language generation. Aims to generate entire sequences in one forward pass, inspired by consistency models from image generation.
- **Quality:** UNKNOWN (very recent, no extensive benchmarks)
- **Efficiency:** Theoretical 100x+ faster (one step vs hundreds)
- **Maturity:** Theoretical / early proof-of-concept
- **Verdict:** PROMISING (revolutionary IF quality gap closes)

#### v4-R1-A1-F6: Diffusion LMs are Provably Optimal Parallel Samplers
- **Source:** NeurIPS 2025
- **Summary:** Formal proof that diffusion models are provably optimal for parallel token generation under certain conditions.
- **Quality:** N/A (theoretical proof)
- **Maturity:** Theoretical
- **Verdict:** PROMISING (strong theoretical foundation)

### SPEEDUP TRICKS (Still AR Under the Hood)

#### v4-R1-A1-F7: Medusa — Multi-Head Speculative Decoding
- **Source:** ICML 2024 + production 2025
- **Summary:** Extra prediction heads on AR model guess future tokens. Base model verifies. 2-2.3x speedup with ZERO quality loss. Deployed on AWS SageMaker.
- **Quality:** SAME (lossless verification)
- **Efficiency:** 2-2.3x faster
- **Maturity:** Production-ready
- **Verdict:** PROMISING (speedup trick, not paradigm shift)

#### v4-R1-A1-F8: PEARL — Adaptive Speculative Decoding
- **Source:** ICLR 2025, arXiv 2408.11850
- **Summary:** Adaptive draft length for speculative decoding. 2.8x faster on HumanEval (H100).
- **Quality:** SAME (lossless)
- **Maturity:** Production-ready
- **Verdict:** PROMISING (best-in-class speculative, still AR trick)

### TRAINING OBJECTIVE IMPROVEMENTS

#### v4-R1-A1-F9: Meta Multi-Token Prediction
- **Source:** Meta, 2024 + 2025 follow-ups
- **Summary:** Train to predict multiple future tokens simultaneously. 4-6% quality improvement. Still generates autoregressively at inference.
- **Quality:** BETTER than standard AR training
- **Efficiency:** 1.3x faster training, same inference
- **Maturity:** Production-ready (Meta uses in production)
- **Verdict:** PROMISING (training improvement, not generation paradigm shift)

### ADDITIONAL FINDINGS

#### v4-R1-A1-F10: SDTT Self-Distillation Through Time
- **Source:** ICLR 2025, arXiv 2410.21035
- **Summary:** Reduces diffusion steps from 50 to 4 via distillation. Makes diffusion LMs practical.
- **Quality:** CLOSE after distillation
- **Efficiency:** 12x faster than baseline diffusion (still slower than AR)
- **Verdict:** NEEDS-MORE-WORK

#### v4-R1-A1-F11: DLMs Struggle with Truly Parallel Decoding
- **Source:** arXiv 2602.23225v1, February 2026
- **Summary:** Analysis reveals "parallel" DLMs frequently converge to AR-like sequential dynamics. Fundamental tension between parallel generation and language's sequential nature.
- **Verdict:** NEEDS-MORE-WORK (important caveat for all diffusion approaches)

### REJECTED

#### v4-R1-A1-F12: CMLM (Conditional Masked Language Model)
- **Quality:** 15-20% BLEU gap in MT, worse in general text
- **Verdict:** REJECTED (quality gap too large)

## Key Insights — Aspect 1

1. **Diffusion LMs have CLOSED the quality gap** — LLaDA matches LLaMA3, Mercury 2 is production-deployed at 5x speed
2. **Speed advantage is REAL** — Mercury 2: 1009 tok/s vs ~200 tok/s for AR
3. **Scaling favors diffusion in data-constrained settings** — more params, less data needed
4. **Fundamental tension exists** — DLMs may collapse to AR-like behavior (needs investigation)
5. **Speculative decoding is a trick, not a paradigm shift** — 2-3x speedup but still AR
6. **Multi-token prediction improves training but not inference** — complementary technique
7. **One-step generation is the holy grail** — if achieved, 100x+ speedup possible

## For Our Project

**CRITICAL OPPORTUNITY:** Diffusion + Ternary could be a killer combination:
- Diffusion provides 5x inference speed via parallel generation
- Ternary provides memory efficiency (add/subtract instead of matmul)
- Both are production-validated independently
- Combined: potentially 5x speed × memory efficiency = unprecedented efficiency

**Open questions for math-paradigm-breakers:**
- How does masked diffusion interact with ternary quantization?
- Can SSM state be used as diffusion denoising backbone (instead of bidirectional transformer)?
- What is the theoretical lower bound on diffusion steps for text?
