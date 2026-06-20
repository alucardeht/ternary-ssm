---
name: VISAO DO PROJETO — modelo pessoal, treinavel, vivo
description: O objetivo final NAO e so importar e rodar. E ter um modelo PESSOAL que o usuario TREINA incrementalmente, ensina skills novas, e que evolui com ele. Nada de LoRA ou gambiarras. Treino REAL. Ler SEMPRE antes de decisoes de arquitetura.
type: project
---

## O MODELO PESSOAL

O usuario quer um modelo que:
1. **E SEU** — roda local, ele controla tudo
2. **TREINA DE VERDADE** — nao LoRA, nao adapter, nao gambiarra. Treino real dos pesos.
3. **APRENDE INCREMENTALMENTE** — quer ensinar Rust? Faz dataset, treina uns minutos, pronto.
4. **NAO RESETA** — diferente de LLMs fechados (Claude, GPT) que perdem contexto, o modelo pessoal RETÉM o que aprendeu
5. **E RAPIDO E LEVE** — roda em consumer GPU (12GB VRAM / 32GB RAM)
6. **E FLEXIVEL** — serve pra conversar, codar, analisar, qualquer coisa que o usuario precise

## POR QUE ISSO IMPORTA PRA ARQUITETURA

Qualquer arquitetura nova PRECISA suportar:
- **Importacao** (converter transformer existente pra comecar com qualidade)
- **Treino incremental** (continuar treinando apos importacao, sem destruir o que ja sabe)
- **Pesos ternarios** (pra caber em hardware consumer)
- **Inferencia O(1)** (sem KV cache crescente)

Se a arquitetura so suporta inferencia mas nao treino incremental = INUTIL.
Se suporta treino mas precisa de cluster = INUTIL.
Se treina mas esquece o que sabia (catastrophic forgetting) = INUTIL.

## O DEFEITO DO PARADIGMA ATUAL (motivacao do projeto)

O usuario usa Claude e sofre com:
- Context reset (/compact) perde tudo
- Precisa de "memorias" como gambiarra pra reter informacao
- Nao pode treinar o modelo pro que precisa
- Modelo e fechado, nao e dele

O projeto existe pra RESOLVER isso: modelo proprio, treinavel, que RETÉM.

## PRIORIDADES (o que fazer primeiro vs depois)

O usuario reconhece que tentar resolver TUDO de uma vez pode dar merda.

**Agora (prioridade):**
- Importacao 1:1 de transformer
- Inferencia eficiente (O(1), ternario, consumer GPU)

**Depois (otimizacao):**
- Treino incremental eficiente
- Anti-catastrophic-forgetting
- Dataset generation pipeline

A arquitetura deve ser COMPATIVEL com treino incremental desde o design, mesmo que a implementacao venha depois.
