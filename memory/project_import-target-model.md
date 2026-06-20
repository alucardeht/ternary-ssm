---
name: Modelo alvo pra importacao — Qwen3-32B
description: Modelo escolhido como alvo principal de importacao. Qwen3-32B Apache 2.0, 8.2GB ternario, cabe em 12GB VRAM. Fallback maior: Llama 3.3 70B (17.5GB ternario, 32GB RAM).
type: project
---

## Modelo Alvo: Qwen3-32B

- **Params:** 32.8B
- **Licenca:** Apache 2.0 (totalmente livre)
- **Arquitetura:** 64 layers, GQA 64/8, hybrid thinking mode
- **Qualidade:** equivalente ao Qwen2.5-72B
- **Treinado em:** 36T tokens

## Tamanhos apos conversao ternaria

| Modelo | FP16 | Ternario (2-bit) | Cabe em |
|--------|------|-------------------|---------|
| Qwen3-32B | 65.6 GB | 8.2 GB | 12 GB VRAM (direto) |
| Llama 3.3 70B | 140 GB | 17.5 GB | 32 GB RAM (streaming) |
| Qwen3-235B-A22B (MoE) | ~470 GB | ~5.5 GB ativos | 12 GB VRAM |

## Pra teste inicial
GPT-2 small (124M) — ja baixado e em cache. Validar formula aqui antes de escalar.

## Pesquisa: `.workspace/analysis/r220-large-model-candidates.md`
