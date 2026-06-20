---
name: Import Status — transformer→SSM conversion research state
description: Estado completo da pesquisa de importação de modelos. Ler IMEDIATAMENTE após /compact se trabalhando em importação.
type: project
---

## OBJETIVO: importar modelo fp32 transformer → nosso SSM ternário, 1:1 perfeito, rápido

## O QUE FOI TESTADO E NÃO FUNCIONA (não repetir)
- OLS linear (W@x→Δ): teto ~50× PPL, irredutível por qualquer feature/ponderação
- NFOLS [x, SiLU(x), x/||x||]: PIORA PPL (110× vs 45× do linear)
- GLS (logit-weighted OLS): ganho marginal 1.5×
- Sequential OLS (usar SSM output como input): PIORA PPL
- MBOLS (mais dados de calibração): satura em M=20, mais dados NÃO ajudam
- Estabilização espectral: PIORA (eigenvalues >1 são legítimos)
- Normalização de target: converge per-layer mas QUEBRA PPL end-to-end (66M!)
- LR adaptativo por ||delta||: piora

## O QUE FUNCIONA MAS NÃO É 1:1
- GD no SSM completo: cos=0.997 em 200 steps, PPL variável (seed-dependent)
- OLS init + GD: potencialmente 10-15 steps com preconditioning (não testado com PPL)

## ROOT CAUSES IDENTIFICADOS
1. **Condicionamento κ(H_k)**: layers tardias têm κ alto → GD lento
2. **||delta|| cresce 7× com profundidade**: layer 11 é a mais difícil
3. **Composição de erro**: 1.7%/layer × 12 layers → exponencial no PPL
4. **Cos ≠ PPL**: cos=0.995 nos hidden states = 45× PPL (amplificação exponencial pelo lm_head)

## O GAP FUNDAMENTAL
O teto analítico (OLS) é ~50× PPL. O gap vem das NÃO-LINEARIDADES:
- softmax (attention), SiLU (FFN/gate), LayerNorm (divisão por norma)
NENHUMA feature analítica testada fecha esse gap.

## O QUE FALTA: A FÓRMULA
Precisa de uma fórmula NOVA que capture as não-linearidades do transformer
de forma closed-form. Ninguém derivou isso ainda. É O TRABALHO A FAZER.

## DOCS DE REFERÊNCIA
- `.workspace/research/r206-lossless-import-discussion.md` — transformer IS an SSM (KV cache)
- `.workspace/research/r207-derive-conversion-formula.md` — 6 fórmulas derivadas (3 corretas, 3 com erros)
- `.workspace/research/r209-corrected-conversion-algorithm.md` — algoritmo corrigido 12 steps
- `.workspace/research/r211-fast-import-breakthrough-discussion.md` — 6 agentes, 5 alternativas
- `.workspace/research/r212-in-proj-conflict-solution.md` — V/K/Q placement resolution
- `.workspace/research/r213-perfect-import-discussion.md` — feature maps + lm_head correction
- `.workspace/research/r214-import-core-problem-discussion.md` — MBOLS/Sequential testados
- `.workspace/research/r215-fast-nonlinear-import-discussion.md` — NFOLS, PCPI, Two-Phase
- `.workspace/research/r216-fastest-gd-import-discussion.md` — curvatura + tempo pra 120B
- `.workspace/research/r217-import-deeper-thinking-discussion.md` — preconditioning + composição

## SCRIPT DE TESTE
- `src/model_v11/experiments/r216_import_test.py` — GD layer-by-layer com GPT-2 small
- GPT-2 já baixado e em cache HuggingFace

## NÚMEROS REAIS MEDIDOS
- GPT-2 PPL: ~81
- OLS analítico: ~5000 PPL (50×)
- GD 50 steps LR=3e-4: PPL 1.6K-64K (variável por seed)
- GD 200 steps: cos=0.997 per layer, PPL não medido com seed fixa
