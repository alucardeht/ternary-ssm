# v12 Runtime Auto-Config — Macro Plan

## Objetivo
Eliminar TODA config manual. O sistema detecta hardware, calcula o ótimo, e configura sozinho.
Nunca trava, nunca crasheia, escala de 4GB mobile a 80GB A100.

## Fase 1 — Hardware Detection (runtime/hardware.c)
- Detectar CPU: cores, cache L1/L2/L3 sizes, SIMD support (AVX2, AVX-512, NEON)
- Detectar RAM: total, livre
- Detectar GPU: presente? qual? VRAM total/livre? Vulkan support?
- Detectar NPU: presente? inference-only?
- Tudo em runtime, zero hardcoded. Retorna struct HardwareProfile.

## Fase 2 — Memory Budget Calculator (runtime/budget.c)
- Input: HardwareProfile + model dimensions (n_layers, d_model, d_state, vocab, seq_len)
- Calcula: tamanho de cada buffer (model, cache, optimizer, gradients)
- Decide: o que cabe em VRAM, o que fica em DDR5
- Calcula: max batch_size que cabe sem exceder 85% de VRAM E 85% de RAM
- Output: struct MemoryPlan (placement de cada buffer, batch_size, etc.)

## Fase 3 — Operation Placement (runtime/placement.c)
- Input: HardwareProfile + MemoryPlan
- Pra cada operação (embed, matvec, scan, optimizer, etc.):
  - Calcula custo em CPU (cores × SIMD) vs GPU (Vulkan compute)
  - Decide placement ótimo
- Output: struct ExecutionPlan (enum per operation: CPU/GPU/HYBRID)

## Fase 4 — Vulkan Compute Backend (runtime/vulkan_backend.c)
- Init Vulkan device, allocate VRAM buffers
- Shaders: ternary_matvec.comp, ssm_scan.comp, sampled_softmax.comp
- Upload/download buffers
- Async dispatch + CPU overlap

## Fase 5 — CPU Threading (runtime/cpu_threads.c)
- OpenMP ou pthreads pool
- Thread count = min(CPU_cores - 2, max_useful_threads)
- Parallel loops pra: embed lookup, optimizer, gradient accum

## Fase 6 — Unified Training Loop (train.c refactor)
- train_step usa ExecutionPlan pra dispatch cada operação
- Se GPU: dispatch Vulkan compute
- Se CPU: dispatch OpenMP parallel
- Se HYBRID: overlap CPU+GPU

## Fase 7 — Smoke Test vs Production Mode
- Programa detecta argv flag: --smoke ou --production (default)
- Smoke: batch=1, steps=10, skip checkpoint, minimal logging
- Production: auto-flex everything, full pipeline

## Modelo de Dimensões: RUNTIME, não compile-time
- V12_D_MODEL, V12_D_STATE, V12_N_LAYERS etc viram campos de uma struct ModelConfig
- Lidos de arquivo config.json OU inferidos do checkpoint
- ZERO #define pra dimensões — tudo runtime
- Structs usam ponteiros (já fazem), sizes passados como parâmetros

## Ordem de Implementação
1. Hardware detection (independente, pode testar isolado)
2. ModelConfig runtime (eliminar #defines, usar struct)
3. Memory budget calculator
4. CPU threading (OpenMP, ganho imediato de 5-10×)
5. Vulkan backend (maior esforço, maior ganho)
6. Operation placement
7. Smoke/production modes
8. Testes end-to-end

## Estimativa
- Fases 1-3: ~300 linhas C cada, testáveis isoladamente
- Fase 4 (Vulkan): ~1000 linhas (init + 3 shaders + dispatch)
- Fase 5 (threads): ~50 linhas (pragmas OpenMP)
- Fase 6-7: refactor de train.c (~200 linhas mudadas)
