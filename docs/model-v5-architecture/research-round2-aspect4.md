# Research Round 2, Aspect 4: Data Efficiency

## Wave 1 — Web Research (2 researchers, Mar 2026)

### Theme A: Data Augmentation, Curriculum & Sample Efficiency

#### R2-A4-F1: OPSD — On-Policy Self-Distillation
- **Source:** Zhao et al., "Self-Distilled Reasoner", arXiv 2026
- **Summary:** Single model serves as both teacher (with privileged info — ground truth) and student (without). Token-level distribution matching provides dense supervision. 4-8x fewer tokens than GRPO while achieving better performance. 2k tokens (OPSD) matches 16k tokens (GRPO).
- **Data efficiency gain:** 4-8x data reduction for equivalent quality
- **Applicability:** Dense token-level supervision ideal for ternary training where gradient signal is already sparse from STE. Self-distillation eliminates need for separate teacher model (memory efficient).
- **Verdict:** PROMISING — highest data efficiency gain, clean mathematical formulation

#### R2-A4-F2: SWIFT — Token-Level Importance Weighting
- **Source:** NeurIPS 2025, "Token-Level Self-Play with Importance-Aware Guidance"
- **Summary:** Assigns per-token importance weights estimated from stronger teacher model. Not all tokens are equally valuable for learning. Surpasses both SFT and uniform DPO.
- **Data efficiency gain:** Significant improvement by focusing learning on high-value tokens
- **Applicability:** Can weight tokens based on contribution to ternary quantization error. Focuses gradient updates on positions where {-1,0,+1} decision boundaries are most uncertain.
- **Verdict:** PROMISING — natural fit for ternary training

#### R2-A4-F3: TernaryCLIP Knowledge Distillation
- **Source:** Zhang et al., "TernaryCLIP", arXiv 2025
- **Summary:** Integrates ternary quantization with knowledge distillation. Teacher FP16 logits guide ternary student. KD provides soft targets that smooth STE gradient discontinuities.
- **Data efficiency gain:** Significant quality improvement at same data budget vs direct ternary training
- **Applicability:** Proven technique specifically for ternary weights. Can use FP16 teacher to guide ternary student.
- **Verdict:** PROMISING — direct evidence of KD working for ternary weights

#### R2-A4-F4: AdaKD / AdaSPEC — Adaptive Token Filtering for Distillation
- **Source:** EmergentMind 2025 survey, multiple papers
- **Summary:** Token-wise KL divergence with adaptive temperature and dynamic token focus. AdaKD: +0.1-2 ROUGE-L. AdaSPEC: +2-15 points acceptance rate for speculative decoding. LeaF: gradient-guided token pruning, +2.4 points on math.
- **Data efficiency gain:** Prevents capacity dilution in small models by filtering which tokens to distill from
- **Applicability:** Adaptive filtering prevents ternary model from wasting capacity on uninformative tokens. Can prune tokens with low sensitivity to ternary quantization.
- **Verdict:** PROMISING — multiple proven variants for LLM distillation

#### R2-A4-F5: Curriculum by Quantization Difficulty
- **Source:** Multiple 2025 papers on data efficiency + quantization surveys
- **Summary:** Order training samples from easy-to-quantize to hard-to-quantize. Sequences with activations near ternary thresholds are "hard". Start with sequences that quantize cleanly, gradually introduce ambiguous cases.
- **Data efficiency gain:** Faster convergence, reduced early-training instability
- **Applicability:** Directly addresses known ternary training instability. Reduces early quantization errors.
- **Verdict:** NEEDS-MORE-WORK — recent evidence (B6, B7) shows curriculum has NO benefit for NLP transformers. SSM might differ but unproven.

#### R2-A4-F6: Data Augmentation for Quantized Models
- **Source:** Multiple 2025 IEEE/Springer surveys on binary/ternary networks
- **Summary:** Data augmentation essential for low-bit quantization performance. Specific recommendations: mixup at token embedding level, noise injection to simulate quantization errors, multi-task learning (next-token + masked LM) for denser signal.
- **Data efficiency gain:** Established best practice for quantized models. Richer training signal per example.
- **Applicability:** Mixup at embedding level creates synthetic intermediate examples. Noise injection pre-conditions model for quantization errors. Multi-task provides denser signal.
- **Verdict:** PROMISING — established best practice, low implementation risk

#### R2-A4-F7: Multi-Token Prediction Self-Distillation
- **Source:** arXiv 2026, "Multi-Token Prediction via Self-Distillation"
- **Summary:** Model learns to predict multiple future tokens simultaneously. Better utilization of each training token by extracting multi-step predictions.
- **Data efficiency gain:** Higher data utilization per token
- **Applicability:** Increases compute per token (multiple forward passes). Unclear if benefit outweighs cost for ternary models. May conflict with SSM sequential state updates.
- **Verdict:** NEEDS-MORE-WORK — interesting but unclear cost/benefit for ternary SSM

### Theme B: Data Selection, Filtering & Synthetic Data

#### R2-A4-F8: ProX — Programming Every Example
- **Source:** Fan Zhou et al., "Programming Every Example", ICLR 2025
- **Summary:** Uses small LMs (0.3B params) to generate and execute fine-grained refinement operations on EACH training example (string normalization, structural fixes). Treats data refinement as programming. ProX data outperforms original by +2% across 10 benchmarks. Domain-specific: +7.6% Mistral-7B, +14.6% Llama-2-7B within 10B tokens (comparable to Llemma-7B trained on 200B tokens = 20x data reduction).
- **Data efficiency gain:** 10-20x data reduction for domain-specific tasks. +2% general.
- **Applicability:** Operates at data preprocessing layer, independent of model architecture. Ternary models benefit MORE from clean data since they have less capacity to memorize noise. Small refiner models (0.3B) run on CPU.
- **Verdict:** PROMISING — highest potential data reduction, architecture-independent

#### R2-A4-F9: Synthetic Data Flywheel
- **Source:** Invisible Technologies blog, Dec 2025
- **Summary:** Curate small high-quality human data → use LLMs to generate synthetic variants (rephrasings, edge cases) → humans filter/validate → train on hybrid corpus. 10k human → 100k synthetic → filter → 50k hybrid = equivalent to 100k fully human-labeled.
- **Data efficiency gain:** 5-10x effective data expansion from small seed set
- **Applicability:** Use open-source LLMs (Llama-3-8B) to generate synthetic text, filter for quality (perplexity < threshold, no repetition). Train on 80% real + 20% synthetic.
- **Risk:** Cap synthetic at 20% of total corpus. Validate ALL experiments on held-out REAL data. Ternary models have limited capacity — low-quality synthetic data hurts MORE than FP16.
- **Verdict:** PROMISING — with strict quality safeguards

#### R2-A4-F10: Typicality-Based Data Selection
- **Source:** Lightly.ai platform, 2025
- **Summary:** Balances density-aware sampling (representative examples) with diversity (edge cases) using embedding space. Combines with uncertainty sampling in active learning loop. 1000 selected images improved recall +32%, F1 +10% vs random.
- **Data efficiency gain:** 3-5x fewer samples for equivalent performance
- **Applicability:** Language modeling lacks labeling oracle. Could repurpose: use ternary SSM's own embeddings to select diverse pre-training documents, avoid redundant dense clusters.
- **Verdict:** NEEDS-MORE-WORK — explore embedding diversity for document selection, skip uncertainty-based selection

#### R2-A4-F11: Density-Aware Core-Set Selection
- **Source:** ACM KDD 2022, applied in production 2025
- **Summary:** k-center algorithms in embedding space with density weighting to select diverse core-set covering data distribution. 40-50% data reduction for same accuracy vs random sampling.
- **Data efficiency gain:** 2-3x fewer samples needed. 40-50% data reduction.
- **Applicability:** Repurpose for document-level diversity selection at data curation stage. Use TF-IDF or frozen embedding model for cheap document embedding.
- **Verdict:** NEEDS-MORE-WORK — consider at data curation stage, not during training

#### R2-A4-F12: Multi-Epoch Progressive Quantization
- **Source:** Emergent Mind synthesis, Jan 2026
- **Summary:** CBM (Curriculum by Masking) uses repeated schedules across epochs with different random seeds to prevent catastrophic forgetting. Soft-to-hard quantization across epochs: Epoch 1 = soft (tau=1.0), Epoch 2 = medium (tau=0.5), Epoch 3+ = hard (tau=0.1).
- **Data efficiency gain:** Prevents forgetting on small datasets, enables longer training without overfitting.
- **Applicability:** If training on small datasets (<10M tokens), multi-epoch with progressive ternary regularization may prevent weight collapse to trivial solutions (all zeros).
- **Verdict:** NEEDS-MORE-WORK — test only if training on small datasets

## Key Insights — Aspect 4 Wave 1

1. **Token-level techniques dominate 2025-2026**: OPSD, SWIFT, AdaKD all focus on per-token objectives rather than sequence-level
2. **Self-distillation eliminates teacher overhead**: OPSD achieves 4-8x data efficiency without separate teacher model
3. **Data QUALITY > data QUANTITY for ternary**: ProX shows 10-20x data reduction via quality. Ternary models have less capacity to memorize noise — clean data matters more.
4. **Curriculum learning debunked for NLP**: Multiple 2026 papers show NO benefit for transformer NLP tasks. May differ for SSM but unproven.
5. **Synthetic data viable with safeguards**: 5-10x expansion possible but cap at 20%, validate on real data only
6. **Knowledge distillation proven for ternary**: TernaryCLIP provides direct evidence that KD + ternary works

## For Math-Paradigm-Breakers (Wave 2)

1. Can we derive the information-theoretic minimum data requirement for a ternary model vs continuous?
2. Is there a principled way to weight training tokens by their "ternary information content" (how much they tell us about where thresholds should be)?
3. Can ternary structure be exploited for more efficient self-distillation (discrete teacher → discrete student)?
4. What is the optimal data mixing strategy from first principles for a ternary SSM?
5. Can we formalize the relationship between data quality and ternary capacity utilization?
