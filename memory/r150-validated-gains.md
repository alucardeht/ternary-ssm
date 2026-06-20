---
name: R150 validated gains
description: Lazy cache refresh gives 2-3× FP speedup — relevant because FP runs ~2300× for 7B real training
type: project
---

**R150 validated (2026-03-18):**
- Lazy cache refresh: 2-3× FP speedup. FP = 29 min per 26M tokens. Para 60B tokens = ~2300 runs. Economia de ~1500-2300 horas.
- **Why:** Cache PCA-INT8 já existe (R120). Medir staleness ||ΔH_{k+1}||/||H_{k+1}|| após cada stage, refresh lazy cada M stages.
- **How to apply:** Implementar após pesquisa convergir em paradigma de treino final.

**R150 mortos:**
- 11B tokens suficiente: confunde capacidade com sample complexity
- Closed-form solve 123,000×: target Δ_k requer backprop (circular), só 15% params
- Gibbs/MCMC 6.4M tokens: mixing time exponencial

**Problema real:** 495 dias em consumer GPU. 2× speedup = 247 dias. Paradigma FP+SBP é insuficiente pra 7B real em horas/dias.
