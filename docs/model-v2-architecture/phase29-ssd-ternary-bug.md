# Phase 29: SSD + Ternary Quantization — Gradient Vanishing Bug

**Date:** 2026-02-26
**Status:** CRITICAL - Blocks all SSD training with ternary
**Severity:** Catastrophic (model diverges by step 15)

## Symptom

Training with `scale_50m_ssd` config:
- Batch: 32, LR: 6e-4, synthetic data
- Steps 1-7: Normal training (loss decreasing, large gradients)
- Step 7: GradNorm = 240.2758 (HUGE, clipped to 1.0 by clip_grad_norm)
- Step 8: **GradNorm = 0.0000 (complete vanishing)**
- Steps 8-11: Loss frozen at 9.0109 (no weight updates)
- Steps 12-15: Loss = NaN (numerical instability)

## Investigation Results

### Tests Performed

| Test | Result | Conclusion |
|------|--------|-----------|
| Disabled FLA Triton kernels | Same issue | Not FLA |
| Used pure Python SSD scan | Same issue | Not SSD-specific kernel |
| Fixed FusedSegSumExp backward | Same issue | Not custom autograd |
| Disabled ternary weight cache | Same issue | Not gradient cache poisoning |
| Disabled Liger fused CE | Same issue | Not loss computation |
| Tested SelectiveSSMSSD gradient flow alone | WORKS (grads flow) | SSD alone is OK |
| Tested with FP16 (non-ternary) | Unknown | Need to test |

### Code Locations

- **SSD scan:** `src/model_v2/components/ssm.py::SelectiveSSMSSD._ssd_scan_chunked()`
- **Ternary quantize:** `src/model_v2/components/ternary.py::ternary_quantize_ste()`
- **Training loop:** `src/model_v2/train.py::train()`
- **Config:** `src/model_v2/config.py::config_scale_50m_ssd()`

### Critical Observations

1. **Huge gradients at step 7:** GradNorm = 240 before clipping
   - This is 240x larger than expected
   - Suggests gradient explosion in ternary weights
   - Clipped to 1.0 by torch.nn.utils.clip_grad_norm_

2. **Weight update consequence:**
   - The lr=6e-4, momentum update with grad=1.0 norm updates all weights significantly
   - Next forward pass (step 8) produces zero gradients
   - Only explanation: weights reach a degenerate quantization state

3. **Loss becomes constant:**
   - Loss = 9.0109 for steps 8-11
   - Same loss across all 4 steps with different random input batches
   - Implies: either model output is frozen OR gradients are completely blocked

4. **Gradient flow isolation test:**
   - Single SelectiveSSMSSD layer forwards + backwards: gradients flow fine
   - But in full model context: gradients vanish
   - Suggests interaction between layers or loss computation

## Hypothesis

**Degenerate Ternary State Hypothesis:**

After the massive gradient update at step 7, the shadow weights enter a state where:
1. Most weights are already quantized (very close to -1, 0, or +1)
2. The quantization operation becomes approximately identity: Q(w) ≈ w
3. The STE formula: w + (Q(w) - w).detach() = w + 0.detach() = w
4. No gradient flows back because the detached difference is zero
5. All parameters receive zero gradients

**Why this happens:**
- Ternary quantization has binary support (-1, 0, +1)
- With 89% ternary + ternary group sizes, weights quickly converge to discrete values
- The threshold-based quantization is harsh: weights below threshold → 0
- Once weights are ≤ threshold, tiny gradients can't move them out

## Proposed Fixes (To Test)

1. **Gradient warm-up:** Start with lower ternary threshold, gradually increase
2. **Stochastic rounding:** Add noise to quantization to prevent early stagnation
3. **Momentum-based shadowing:** Keep track of gradient momentum separately from weight value
4. **FP32 shadow weights:** Store FP32 copy alongside quantized, prevent premature discretization
5. **Progressive ternary:** Don't quantize until step 50-100, then gradually enforce it
6. **Larger group size:** Allow more degrees of freedom per group
7. **Disable ternary early:** First 50 steps full precision, then quantize
8. **Different optimizer:** Try SGD instead of AdamW to reduce momentum effects

## Immediate Action Items

- [ ] Confirm issue with FP16 non-ternary baseline (does it converge?)
- [ ] Test with `scale_50m` (50M FP16, no ternary) - should converge fine
- [ ] Instrument forward pass to check gradient flow at each layer at step 7-8
- [ ] Print ternary weight distributions at steps 1, 7, 8
- [ ] Check if `torch.nn.utils.clip_grad_norm_` is the culprit (try without clipping)
- [ ] Add debug logging to STE backward pass

## Files Modified

- `src/model_v2/components/ssm.py`: Fixed FusedSegSumExp backward (applied mask)
- `src/model_v2/components/ternary.py`: Disabled weight cache

## Next Steps

This is a fundamental architectural issue with SSD + ternary combination. The huge gradients suggest the loss landscape is unstable when quantization is applied. Needs systematic investigation of:
1. Gradient magnitudes at each layer
2. Weight discretization timeline
3. Interaction between SSD state dimension and ternary quantization
4. Whether same issue occurs with non-SSD layers

**Priority:** HIGH - blocks Phase 29 progress
