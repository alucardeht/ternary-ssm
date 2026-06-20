# Expected Differences: v11 Final → v12 C

Every known difference with classification and reason.

## INTENTIONAL (by design, with research reference)

| Feature | v11 Final | v12 C | Reason | Ref |
|---------|-----------|-------|--------|-----|
| Language | Python/PyTorch | C/Vulkan | Migration target | — |
| GPU backend | CUDA | Vulkan | Vendor-neutral, no NVIDIA lock-in | User decision |
| Attention (SWA) | SSM+SWA hybrid | SSM pure | Multiply-free constraint | User decision |
| Gradient filter | D4 sign count | EMA-SPRT | Wald SPRT theorem, provably optimal | R317 |
| Optimizer | Natural gradient LR=0.1 | Factored Adam LR=3e-4 | Different optimizer class | R335 |
| Crystallization tau | 3 | 4.6 | MVUE saturation P(correct)>99% | Paper §5 |
| B init | Eigenvector-aligned | Uniform (Direct-B subsumes) | Dynamic B_eff learns routing | R335 |
| Softmax | Full vocab CE | Sampled K=50 NCE | 660× logits reduction | R332 |
| lm_head | Full vocab updates | Hot/cold K=2000 | 50× late-training speedup | R332 |
| Aux head | lambda=0.1 | lambda=0 (OFF) | Sparse backward incompatible | Implementation note |
| Embedding | Trainable | Frozen | VRAM reduction | v12 decision |

## UNKNOWN (needs verification)

| Feature | v11 Final | v12 C | Risk | Action |
|---------|-----------|-------|------|--------|
| A init timescales | Geometric sweep 128 | 4 fixed (1,8,60,121) cycling | 2-5× slower convergence | CHECK model.c |
| Batch size | 4 | 1 | 4× less data per step | Consider implementing batching |
| LR schedule | Cosine with min=10% | Cosine with min=1% | Floor too low | Verify math |

## BUGS (found, status)

| Bug | Status | Impact |
|-----|--------|--------|
| GPU buffer overflow (vk_buf_y) | FIXED | Was causing NaN/freezes |
| Checkpoint I/O error | NOT FIXED | Can't save/resume overnight |
| stdout buffering | FIXED | Logs not appearing in file |
| aux_head dead code (4.2s/step) | FIXED | Was wasting compute |

## TOKEN COVERAGE COMPARISON

| Run | Steps | Batch | Seq | Tokens Seen | % of 900M |
|-----|-------|-------|-----|-------------|-----------|
| v11 overnight | 55,000 | 4 | 512 | 112M | 12.4% |
| v12 run-001 | 18,000 | 1 | 512 | 9.2M | 1.0% |
| v12 equivalent | 55,000 | 1 | 512 | 28.2M | 3.1% |
| v12 w/ batch=4 | 55,000 | 4 | 512 | 112M | 12.4% |

**Key insight:** v12 run-001 saw 12.2× FEWER tokens than v11 overnight.
To match token coverage: need batch=4 AND 55K steps.
