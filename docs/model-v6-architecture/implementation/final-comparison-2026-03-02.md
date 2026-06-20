# Comparação Final: FP16 vs v5 vs v6

**Data:** 2026-03-02
**Pesquisa:** v6 Round 1-2 completa

---

## Tempos de Treino Projetados (7B parâmetros, 140B tokens)

| Configuração | Tempo | Speedup vs FP16 | VRAM |
|--------------|-------|-----------------|------|
| **FP16** | 280h (12 dias) | 1x | 84 GB |
| **v5 Ternary** | 70h (3 dias) | 4x | 14 GB |
| **v5 + Otimizações** | 47h (2 dias) | 6x | 14 GB |
| **v6 Projetado** | 7h (0.3 dias) | **40x** | 7 GB |

---

## Como chegamos em 40x?

### Base v5: 4x (70h vs 280h)
- Ternary weights: add/sub ao invés de matmul
- 59% throughput gain

### Adições v6 projetadas:
| Aspecto | Contribuição | Mecanismo |
|---------|-------------|-----------|
| Sign-Only Optimizer | 3.3x | Elimina momentum, 0 bytes state |
| Progressive Freezing | 6x | 80% camadas congeladas |
| Energy Training | 3.3x | Elimina optimizer state |
| Backward Fusion | 2x | Kernels fundidos |
| Data Pipeline | 1.5x | Length-sort batching |
| Hybrid Precision | 2x | FP16 → INT8 por camada |

**Combinado (conservativo): 10-15x sobre v5**
- v5: 70h → v6: 5-7h

---

## Progressão de VRAM

| Configuração | VRAM | Redução |
|--------------|------|---------|
| FP16 (Adam) | 84 GB | baseline |
| v5 Ternary | 14 GB | -83% |
| v6 (Energy+Sign) | 7 GB | -92% |

---

## Habilidades Novas Descobertas

### 1. Ising Model Optimization (IMO)
- Otimizador termodinâmico — pesos como spins
- Sem estado, só temperatura
- Escolha radical: minimização de energia vs gradiente

### 2. Latent State Evolution (LSE)
- Decoupling backward da quantização
- Modelo aprendido prediz transições de peso
- Substitui STE completamente

### 3. Self-Signal Continuous Learning (SSCL)
- Modelo gera seus próprios sinais de treino
- Detecta gaps de conhecimento via loss
- Gera exemplos sintéticos targeting弱点

---

## Top Técnicas Implementáveis (Baseadas em Pesquisa)

| # | Técnica | Aspecto | Speedup Esperado |
|---|---------|---------|------------------|
| 1 | SignSGD with Momentum | 5 | 3.3x |
| 2 | AutoFreeze (gradient-norm) | 3 | 6x (80% frozen) |
| 3 | Length-Sorted Batching | 4 | 1.5x |
| 4 | Fused STE Backward | 2 | 2x |
| 5 | Layer-wise Precision | 6 | 2x |
| 6 | GAST-Q Threshold | 1 | 1.3x |

---

## Conclusão

| Métrica | FP16 | v5 | v6 (projetado) |
|---------|------|----|-----------------|
| Tempo 7B | 280h | 47h | 7h |
| Speedup | 1x | 6x | 40x |
| VRAM | 84GB | 14GB | 7GB |

**Próximo passo:** Implementar as 6 técnicas prioritárias e validar com benchmark real.

---

## Arquivos da Pesquisa

- `wave1-findings.md` - Findings Waves 1-3
- `research-plan.md` - Detalhes técnicos Round 2
- `vision-and-status.md` - Status geral
