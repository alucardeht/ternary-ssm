# v11 Overnight Baseline

## Config
- 108M params, 6 layers, d=1024, d_state=128, vocab=32064
- Joint GD from step 0
- LR=0.1 (MVUE natural gradient)
- 55,000 steps
- Dataset: data/chat_full/train.bin (322M BPE tokens)
- PyTorch + CUDA, RTX 5070
- Batch size: 4 (NOT 1)

## Result
- Final loss: 1.86 nats
- Time: 2.6 hours (~0.17s/step)
- VRAM: 1.76 GB (batch=4, no cache)
- Inference throughput: 379 tok/s (108M model)

## Loss Curve
NOT AVAILABLE — only final loss was recorded.
Loss at step 18K: UNKNOWN (not logged at that granularity).

## Source
docs/model-v11-architecture/FINAL-STATE.md §3.1

## Important Notes
- This used NATURAL GRADIENT (Fisher-rescaled LR), not standard Adam
- Batch size was 4 (v12 C uses batch=1 — 4× less data per step)
- 55K steps × batch=4 × seq=512 = 112M tokens seen (12.4% of 900M dataset)
- v12 run-001 at 18K steps × batch=1 × seq=512 = 9.2M tokens (1% of dataset)
- Token coverage difference: 12.2× more tokens in v11 overnight
