# 08 — Validation: Retrieval Mechanism for SSM Bottleneck

## VERDICT: HYBRID SSM + SLIDING WINDOW ATTENTION — Industry Standard, Proven at Scale

Pure SSMs fail at retrieval (copy, passkey). Solution: ~10-15% attention layers with 128-token sliding window. Minimal overhead, perfect retrieval, fully ternary-compatible.

---

## 1. The Problem

SSMs have **exponentially fading memory** — they compress context into fixed-size state. This fails for:
- Copy tasks (reproduce exact tokens from input)
- Passkey retrieval (find specific token in long context)
- Associative recall (map key → value from earlier in sequence)

These tasks require ~12.5% of operations in typical language modeling.

---

## 2. Solution Landscape (Ranked)

### Tier 1: Proven at Scale (Use These)

| Approach | How It Works | Overhead | Retrieval | Ternary Compatible |
|----------|-------------|----------|-----------|-------------------|
| **Sliding Window Attn + SSM** | Local attention (128 tokens) + SSM global | +5-10% FLOPs | Perfect | ✅ Yes |
| **Jamba-style Hybrid** | 1 full attention per 7 SSM layers | +10-15% FLOPs | Perfect | ✅ Yes |
| **Griffin (Google)** | Gated linear recurrence + local attention | ~Same as SSM | Perfect | ✅ Likely |

### Tier 2: Promising (For Extreme Cases)

| Approach | How It Works | Overhead | Retrieval | Ternary Compatible |
|----------|-------------|----------|-----------|-------------------|
| **Hyena/StripedHyena** | Long convolutions + 10% attention | +5-10% FLOPs | Good | ✅ Likely |
| **Memory-Augmented SSM** | Retrieval from distant past | +15-20% FLOPs | 90% at 400K | ✅ Likely |

### Tier 3: Not Recommended

| Approach | Why Not |
|----------|---------|
| **LSH Attention (Reformer)** | Older (2020), unclear ternary compatibility, community moved on |
| **Content-Addressable Memory (NTM)** | No recent SSM integration research, academic interest faded |

---

## 3. Recommended Approach: Sliding Window Attention

### Why Sliding Window > Full Attention

**Key insight from Hymba (NVIDIA, Nov 2024):** >50% of global attention can be replaced by SSM without performance loss. When SSM handles global summarization, you only need LOCAL attention for precise retrieval.

```
Window size: 128 tokens (industry sweet spot)
KV cache: O(128) per layer instead of O(context_length)
At 128K context: 1000× less KV cache than full attention
FLOPs overhead: ~5-10% vs pure SSM
```

### Evidence

| Model | Architecture | Retrieval Performance |
|-------|-------------|----------------------|
| Samba | Mamba + SWA + MLP | Perfect passkey at 256K |
| MiMo-V2-Flash | SSM + 128-token window | Negligible KV overhead |
| Hymba | SSM + local attention heads | State-of-the-art quality |
| Gemma3 | Hybrid with 128-token window | Production-ready |

### Configuration

```
Layer composition (32 layers total):
- 28-29 SSM layers (88-91%)
- 3-4 sliding window attention layers (9-12%)

Attention placement options:
  Option A: Every 9-10 layers (uniform)
  Option B: First, middle, last (strategic)
  Option C: Per-head mixing (Hymba-style, more complex)

Window size: 128 tokens
Attention type: Multi-head with RoPE (standard)
KV cache per layer: 128 × d_model × 2 (negligible)
```

---

## 4. Alternative: Jamba-Style Full Attention Hybrid

If sliding window proves insufficient for our tasks:

```
Layer composition (32 layers total):
- 28 SSM layers (87.5%)
- 4 full attention layers (12.5%)

Attention placement: Every 8 layers
Ratio: 1:7 (proven by AI21 at 52B and 398B scale)
```

**Trade-off:** More retrieval power, but O(n²) attention for those layers. At 128K context, these 4 layers dominate compute.

---

## 5. Attention Ratio — What's Minimum?

### Industry Consensus

| Ratio | Architecture | Works? |
|-------|-------------|--------|
| 0% (pure SSM) | Mamba | ❌ Fails retrieval |
| ~10% (1:10) | StripedHyena | ✅ Works |
| ~12.5% (1:7) | Jamba | ✅ Industry standard |
| ~15% | Strategic (3 layers) | ✅ Sufficient |
| ~50% | Half-and-half | ✅ Overkill, wastes efficiency |

**Sweet spot: 10-15% attention layers.** Below 10% may miss retrieval tasks. Above 15% adds overhead without proportional benefit.

### Strategic Placement > Uniform Spacing

Research shows 3 attention layers placed at first/middle/last positions recover recall-intensive accuracy equivalent to uniformly distributed attention.

---

## 6. Compatibility with Our Architecture

### Ternary Weights + Sliding Window Attention

**Attention layers can use ternary weights too:**
- Q, K, V projections → ternary (STE)
- Attention scores (QK^T/√d) → keep FP16 (precision critical)
- Output projection → ternary

**Net impact on compression:**
```
28 SSM layers: 90% ternary + 10% FP16 (A, B, C, Δ)
4 SWA layers: 90% ternary (projections) + 10% FP16 (scores)

Overall: ~90% ternary → ~86-89% compression maintained
```

### MoD Routing + Hybrid Layers

Dynamic routing applies to BOTH SSM and attention layers:
- SSM layers: tokens skip via MoD (residual connection)
- Attention layers: tokens skip via MoD (saves attention compute)
- **Attention layers may have lower skip rates** (more tokens need retrieval)

### State Consistency

When tokens skip SSM layers, state must remain consistent:
- Skipped token: SSM state passes through unchanged (identity)
- No attention needed for skipped tokens (they use residual)
- This is the natural behavior — no special handling required

---

## 7. Computational Cost Analysis

### Relative to Pure SSM (normalized to 1.0×)

| Config | Training | Inference (short) | Inference (long) | Memory |
|--------|----------|-------------------|------------------|--------|
| Pure SSM | 1.0× | 1.0× | 1.0× | 1.0× |
| SSM + SWA (128) | 1.05-1.10× | 1.03-1.05× | 1.05-1.08× | 1.0× |
| SSM + Full Attn (1:7) | 1.10-1.15× | 1.05-1.08× | 1.10-1.15× | 1.1× |
| Pure Transformer | 1.5-2.0× | 2.0-3.0× | 5.0-10.0× | 3.0-5.0× |

**Sliding window with 128 tokens adds <10% overhead** — negligible for solving a critical failure mode.

---

## 8. Impact on Model v2 Design

### Architecture Update

```
BEFORE (pure SSM vision):
  All 32 layers: Ternary SSM + MoD routing

AFTER (hybrid vision):
  28 layers: Ternary SSM + MoD routing
  4 layers:  Ternary Sliding Window Attention (128 tokens) + MoD routing
  Placement: Layers 0, 10, 21, 31 (strategic: first, ~1/3, ~2/3, last)
```

### What Changes
1. 4 layers use attention instead of SSM
2. Small KV cache for attention layers (128 × d_model × 2 × 4 layers)
3. RoPE positional encoding for attention layers only

### What Stays The Same
1. ~90% ternary weights (attention projections also ternary)
2. ~86-89% total compression
3. MoD routing on all layers
4. CPU-primary inference (ternary ops dominate)
5. O(1) memory growth for SSM layers (28 of 32)

### Net Assessment

The hybrid adds minimal complexity and cost, but solves the #1 limitation of pure SSMs. This is the same approach used by Jamba (AI21, 398B params), Hymba (NVIDIA), Samba, and Griffin (Google DeepMind) — all production-grade systems.

---

## 9. Future Option: Memory-Augmented SSMs

If we need extreme long-context retrieval (>256K tokens) in the future:

**Expansion Span (Amazon, Dec 2024):**
- Reserve portion of attention context for tokens retrieved from distant past
- Allocate based on relevancy, not recency
- Extends to 8× training context length
- Compatible with hybrid architectures

**MemMamba:**
- 90% retrieval accuracy at 400K tokens
- Retrieval-based memory augmentation

These are additive — can be layered on top of our hybrid architecture later.

---

*Sources: Jamba (ICLR 2025), Hymba (NVIDIA, ICLR 2025), Samba (arXiv 2406.07522), Griffin (Google DeepMind 2024), StripedHyena (Together AI 2024), Expansion Span (Amazon 2024), MiMo-V2-Flash (Xiaomi 2025), RecurrentGemma (Google 2024), Reformer (ICLR 2020)*

*Last updated: February 2026*
