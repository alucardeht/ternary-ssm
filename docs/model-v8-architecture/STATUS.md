# Model v8 — Status

**Created:** 2026-03-08
**Phase:** Research (R18 — 11 rounds planned)
**Based on:** v7 post-R17 (commit 9903073)

## Estado

v8 está na fase de pesquisa. Nenhum código ainda — src/model_v8/ será criado após identificar breakthrough arquitetural.

**Objetivo da pesquisa v8:**
Encontrar descobertas COMPLEMENTARES ao v7, especialmente:
1. Alternativas ao gradient descent para ternário (além de DSST)
2. Arquiteturas que eliminem o gap de VRAM de treino (28× vs teórico)
3. Paradigmas radicais (não só incremental sobre v7)

## Rounds R18 (planejados)

| Round | Aspecto | Status |
|-------|---------|--------|
| R18-A | Token/Representação | pending |
| R18-B | Input Não-Textual | pending |
| R18-C | Aprendizado Contínuo | pending |
| R18-D | Plasticidade Arquitetural | pending |
| R18-E | Lazy Instantiation | pending |
| R18-F | Sistema 1 vs 2 | pending |
| R18-G | Memória Biológica | pending |
| R18-H | Física Computacional | pending |
| R18-I | Software Ternário Nativo | pending |
| R18-J | Nova Matemática | pending |
| R18-K | Avaliação Multi-Dimensional | pending |

## Arquivos

```
docs/model-v8-architecture/
├── STATUS.md              ← este arquivo
├── implementation/        ← implementation-log.md virá aqui
└── research/              ← findings por aspecto

.workspace/research/v8/
├── r18-concept-brief.md   ← briefing para todos os agentes
├── dedup-registry-v8.txt  ← evitar duplicatas entre rounds
└── R18-*/                 ← outputs de cada round
```
