# Model v7 — Status

## Objetivo: VELOCIDADE DE TREINO

**v6 foi descartada por falha total de proposta:** todo o overhead de C2 (sigma_g, admm_u, n_stable, committed_mask = +9 B/param), HFOTT (+1 forward/step), BITDISTILL (+1 forward/step) adicionou custo sem speedup real. O matmul continuou float. Throughput medido: 9.815 tok/s vs 45.225 tok/s (FP16 v5) — **4,6× mais lento**.

## Base

- **Código ativo:** `src/model_v7/` (cópia limpa de src/model_v5/, o baseline rápido)
- **Baseline de referência:** v5 FP16 = 45.225 tok/s, 886 MB VRAM
- **Target v7:** superar v5 FP16 em throughput de treino

## Foco de Pesquisa v7

1. **Primário — Velocidade de treino** (tok/s, time-to-train)
   - Triton/kernels customizados para SSM
   - Ops fusionadas (forward+backward em um kernel)
   - Compilação eficiente (torch.compile max-autotune)
   - Técnicas que reduzem FLOPs reais, não só "convergência"

2. **Secundário — Eficiência de RAM**
   - Reduzir VRAM de treino abaixo de v5 FP16 (886 MB)
   - Activation offloading, recomputation seletiva

## Estado

- [ ] Rodadas de pesquisa definidas
- [ ] R8 (ou R1-v7): pesquisa focada em velocidade
- [ ] Implementações
- [ ] Benchmarks vs v5 FP16

## Lição de v6 (NÃO REPETIR)

> Técnicas de "convergência" (menos steps) ≠ velocidade de treino (tok/s).
> Nunca confundir os dois. Relatório deve sempre mostrar tok/s medido,
> não apenas "estimativa de convergência".
> NADA que adicione overhead por step sem reduzir FLOPs do matmul.
