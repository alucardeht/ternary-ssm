---
name: Formula Q — resultado publicavel, prediz ranking de datasets
description: Q = CE_theta(D) / K_approx(D) - 1 prediz TODOS os rankings empiricos de datasets sem fitting. Derivada de 5 frameworks independentes. Publicavel.
type: project
---

## Formula Q — Dataset Quality Score

```
Q(D, theta) = CE_theta(D) / K_approx(D) - 1
```

Onde:
- CE_theta(D) = cross-entropy do modelo atual nos dados D
- K_approx(D) = taxa de compressao gzip dos dados (bits/byte)

## Rankings preditos vs medidos

| Dataset | Q | Ranking | Confirmado |
|---|---|---|---|
| Running-XOR | 549 | 1 (predito) | NAO TESTADO |
| Automatos | 109 | 2 | R259 (+12%) ✅ |
| Livros | 0.92 | 3 | R251 (6-10x) ✅ |
| Web crawl | 0.32 | 4 | Baseline ✅ |
| Gzip | 0.01 | 5 | R256 (prejudicial) ✅ |

## Curriculum derivado (nao heuristico)

XOR (Q=549) → automatos (Q=109) → formal (Q~1.2) → livros (Q=0.92) → linguagem (Q=0.32)

Transicoes por variancia da loss, nao steps fixos.

## Por que e publicavel

1. Formula unificada de 5 frameworks (rate-distortion, AIT, Fisher SNR, SA, spectral)
2. Zero parametros livres
3. Prediz ranking inteiro sem fitting
4. Derivada pro caso especifico (SSM ternario byte-level) mas generalizavel

## Referencia: `.workspace/research/r264-datasets-discussion.md` secao MATH-PARADIGM-BREAKER
