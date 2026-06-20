---
name: Paper Q Formula — dados e referências para escrita
description: Tudo que temos para escrever o paper sobre Q = CE/K_approx - 1 como métrica de qualidade de dataset
type: project
---

## Paper: Q — A Parameter-Free Dataset Quality Score for SSMs

### Dados empíricos disponíveis

| Dataset | Q | delta_L medido | Experimento |
|---------|---|---------------|-------------|
| Running-XOR | 549 | 94% reduction (5.55→0.30) | R269 |
| Automatos | 109 | 12% faster than WikiText | R259 |
| Livros | 0.92 | 6-10x vs web baseline | R251 |
| Web crawl | 0.32 | Baseline | Baseline |
| Gzip noise | 0.01 | Prejudicial | R256 |

### Métricas complementares (R269)

- Causal KL = 0.98 (d_state=8) — SSM usa estado recorrente
- Stable rank collapse: 187 → 19 (d_state=4), 121 (d_state=8)
- Probe linear R² = 0.09 — matematicamente necessário (Fourier/XOR)
- d_state=8 ótimo mas d_state=4 também aprende

### Derivação teórica

5 frameworks convergem para Q (seção MATH-PARADIGM-BREAKER de r264):
1. Rate-distortion
2. AIT (Algorithmic Information Theory)
3. Fisher SNR
4. Simulated Annealing
5. Spectral

**Caveat validado:** Na inicialização, CE_θ₀ ≈ constante → Q ≈ 1/K_gzip.
Os "5 frameworks" são parcialmente redundantes na init. Poder real = Q(θ_t) online.

### Originalidade confirmada (R275)

- Nenhuma fórmula CE/K_approx na literatura
- QuRating (ICML 2024), DCLM (NeurIPS 2024), BiMix (ICLR 2025) são diferentes
- XOR como benchmark SSM com lower bound derivável = gap na literatura
- State utilization formalizada (stable_rank + probe) = contribuição original

### Arquivos-fonte dos dados

- `src/model_v11/experiments/r269_running_xor.py` — Running-XOR completo
- `src/model_v11/experiments/r259_automaton_vs_wikitext.py` — Automaton vs WikiText
- `.workspace/research/r264-datasets-discussion.md` — Derivação original de Q
- `.workspace/research/r268-datasets-next-discussion.md` — Validação R268
- `.workspace/research/r275-datasets-next2-discussion.md` — Validação R275

### Papers anteriores (estrutura a seguir)

- `papers/paper1-ternary-ssm/` — LaTeX NeurIPS, generate_plots.py
- `papers/paper2-gradient-estimators/`
- `papers/paper3-efficiency/`

### O que falta testar (nice-to-have, não blocker)

- MLP probe com 8+ neurônios (deve dar R² > 0.9)
- Q magnitude validation (power-law fit)
- Cross-model Q (6M vs 50M)
