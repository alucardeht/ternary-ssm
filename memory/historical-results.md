# Historical Results Archive

## R112-R115 FP GAP CAMPAIGN (2026-03-13)

- R112: Reverse BCD → -59% (REJEITADO — residual stream assimétrico)
- R113: SLS N=50 → +14.8% (surrogates fracos prejudicam)
- R114: SLS N=100 + BCD → **+23.2%** (melhor combo)
- R115: Warm Init α=0.3 → +12% sobre SLS (melhor mod isolada). All Combined → -33.4% (interferem)
- **Pipeline ótimo:** SLS N=100 + Warm Init α=0.3 + BCD c2 ≈ 30-35% gap closure
- **Conclusão:** 65-70% do gap é ESTRUTURAL a treino sequencial. Só escala fecha mais.

---

## R116 FP H_k CACHING (2.59× speedup + 0.046 nats melhor)

- Cache ativações frozen → skip forward redundante. 16.3s→6.3s em n=8.
- Mas cache fp32 é GORDO: 16-32 GB pra 7B → tradeoff ruim (20 GB RAM pra poucos min).
- Arquivo: `src/model_v10/experiments/r116_fp_cached.py`

---

## R117 CPU+GPU HYBRID (2026-03-13)

- Hybrid FP32: qualidade MELHOR (-0.028 nats vs R116), speedup marginal (1.08×)
- Hybrid FP16: +0.010 nats overhead, no speedup
- Hybrid INT2: +0.067 nats → REJEITADO
- **Lição 75:** Device transfer overhead domina em small scale. Hybrid compensa VRAM-bound (7B+).
- **Lição 76:** FP16 cache aceitável (±0.01 nats) pra 2× RAM reduction.
- **Lição 77:** INT2 forward em CPU = 6× overhead esperado. Manter fp32 em CPU, INT2 só storage.
- Arquivo: `src/model_v10/experiments/r117_hybrid_cpu_gpu.py`

---

## R118 ACTIVATION QUANTIZATION (2026-03-13) — ACHADO PRINCIPAL

- FP16 cache: -0.038 nats MELHOR que fp32 (regularização!). 7B=15.62 GB.
- INT8 cache: -0.028 nats MELHOR que fp32. 7B=7.81 GB. **MELHOR TRADEOFF.**
- INT4: +0.302 nats → REJEITADO (dead zone)
- Ternary: +0.180 nats, 7B=1.95 GB → VIÁVEL (teoria do usuário CONFIRMADA)
- Binary: +0.237 nats → lossy demais
- **Lição 78:** Quantization noise = regularizer para frozen cache. INT8 > fp32.
- **Lição 79:** INT4 é dead zone — skip direto INT8→ternary.
- **Lição 80:** Ternary activations viáveis (+0.18 nats, 16× compression). Teoria confirmada.
- Arquivo: `src/model_v10/experiments/r118_activation_quant.py`

---

## R120 ADVANCED CACHE COMPRESSION (2026-03-13) — BREAKTHROUGH

- H_k é LOW-RANK: stable_rank=2-5, rank@90%=17-20/256
- PCA r=20 + INT8: **51× compressão** (50 MB→1 MB em n=8, 31 GB→0.61 GB em 7B)
- Quality: +0.114 nats vs fp32 (aceitável — int8 puro = -0.036)
- norm-ternary: COLAPSOU (NaN stage 8) — REJEITADO
- **Lição 84:** H_k vive em subespaço low-rank → PCA antes de quantizar
- **Lição 85:** Normalizar antes de ternary = instável em FP sequencial
- **Lição 86:** PCA calibration com 20 batches é suficiente
- **7B total VRAM: ~6.27 GB** (INT2 pesos + PCA cache + active layer)
- Arquivo: `src/model_v10/experiments/r120_advanced_cache.py`

---

## R108-R109 (anteriores)

- R108/R108b: QOTA → +1.20 nats pior (Taylor inválido para Δ=O(1))
- R109: TRGL +0.016 (neutro). DGL +0.169 (prejudicial)
- R110: DEQ-FP NOT VIABLE (distance=0.84, L=0.97)

---

## Lições 64-74 (R107-R115)

- 64: Taylor inválido para Δ=O(1)
- 65: TRGL ajuda intermediários, última stage desfaz
- 66: Gap FP é information-theoretic
- 67: DEQ requer L < 0.5, distance < 0.3
- 68: 5 abordagens falharam; escala é único mecanismo confirmado
- 69: Reverse BCD viola assimetria residual
- 70: Qualidade do surrogate é crítica (val > 2.5 → prejudica)
- 71: SLS escala com N_fast (mais steps → melhor)
- 72: Warm Init α=0.3 funciona, α=0.1 = ruído
- 73: Mods sistêmicas NÃO combinam linearmente — intervir cirurgicamente
- 74: Melhor pipeline: SLS N=100 + Warm α=0.3 + BCD c2

---

## R91-R106 Summary

Detalhes completos em commits com tags R91-R106. Resumo:
- Byte-level desaparece com escala
- INT2 validado
- BCD/PGLR abandonado
- FP gap é information-theoretic
