# Transformer Residency Lab

This folder is the isolated implementation area for the T001 dense Transformer
consumer-GPU residency line.

It is separate from the project-native ternary SSM/v12 line. Do not modify
`src/model_v12/` or the main training stack when working here unless the user
explicitly asks to bridge the lines.

## Goal

The near-term goal is not maximum token speed. The near-term goal is to test
whether a dense `Qwen/Qwen3-32B`-class Transformer can be represented below a
12 GB GPU residency envelope while preserving quality closer to Q4 than public
Q2/IQ2.

The current gate stack is:

1. Down-visible MLP residual collapse.
2. Metadata-amortized tile receipts.
3. Fragment-first clause feasibility.
4. Margin-aware logit certificates.
5. MLP gate-cliff certificates.

## First Probe

The first implementation slice is a synthetic tile/certificate probe. It answers:

- Does hidden MLP residual collapse after the down projection?
- Do compressed tile receipts still save bytes after metadata, scales,
  correction, scratch, and launch costs?
- Do logit and MLP-gate certificates reject obvious unsafe cases?
- What coarse Qwen3-32B footprint follows from the current effective bits per
  weight?

The probe now includes positive and negative controls for each initial gate. A
run is not healthy unless safe synthetic cases pass and unsafe synthetic cases
fail.

Run:

```powershell
python scripts/research/t001_transformer_residency_probe.py
```

Outputs go under `.work/jobs/t001-transformer-residency-probe/`.

The current probe is intentionally synthetic. It validates gate wiring and stop
conditions before any model download, captured tile, or GPU kernel work.

The Qwen3-32B projection is a coarse shape-based estimate. It is not measured
VRAM and does not include real allocator behavior, KV cache, kernels, or framework
fragmentation beyond the configured overhead knob.

Current default synthetic projection:

- effective candidate bits per weight: `3.000`
- projected uniform candidate footprint: about `11.65 GiB`
- projected component-stack footprint before Qwen3 `head_dim` correction: about
  `11.03 GiB`
- implementation-corrected component-stack footprint with official
  `head_dim=128`: about `11.95 GiB`
- projected component-stack effective bits before correction: about `2.824`
- implementation-corrected component-stack effective bits: about `2.871`
- implementation-corrected Q4 footprint: about `16.26 GiB`
- projected component-stack savings vs Q4: about `4.18 GiB`
- synthetic decode proxy at `500 GiB/s`: about `49.87 tok/s`
- synthetic decode proxy speedup vs Q4: about `1.416x`
- synthetic decode proxy speedup vs uniform probe: about `1.062x`
- synthetic control checks: `22`
- unexpected controls: `0`

The decode proxy is a memory-bound weight-stream estimate, not real model
throughput. It is controlled by `--synthetic-bandwidth-gib-per-second`,
`--weight-streams-per-token`, and `--kv-cache-gib-per-token`.

Next step: replace synthetic residuals with captured or model-derived MLP tile
residuals while preserving the same pass/fail controls.

## T002 KV Margin Sweep

The first T002 implementation slice adds an opt-in KV-cache margin sweep:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t002-kv-margin-sweep
```

It emits `T002_KV_MARGIN_SWEEP_ACTIVE` and writes a before/after table under
`.work/jobs/t001-transformer-residency-probe/`.

Latest synthetic gate result:

- corrected before: `12.949 GiB` total at 4k FP16 KV, so the corrected Qwen3
  denominator does not fit 12 GiB.
- minimal usable: `11.557 GiB` at 8k INT8 KV with static arena plus light
  tightening.
- strong near-term: `10.739 GiB` at 16k INT4 tiered KV with `1.261 GiB` slack.
- tok/s push proxy: `1.583x` synthetic speedup when exact self-speculative
  verification is layered on the strong 16k memory stack.

This remains a synthetic denominator and wiring gate, not measured Qwen3 VRAM or
real model throughput.

## T002 Real CUDA Benchmark

The second T002 slice measures local CUDA memory bandwidth with real GPU buffers
and reprices the same denominator:

```powershell
python scripts/research/t002_kv_real_benchmark.py --out-dir .work/jobs/t002-kv-real-benchmark/latest --buffer-gib 1.0 --repeats 24 --warmup 4
```

Latest result on the RTX 5070:

- real 1 GiB CUDA copy bandwidth: `514.568 GiB/s`
- real 2 GiB CUDA copy bandwidth repeat: `512.546 GiB/s`
- corrected before proxy: `43.06 tok/s`
- strong 16k INT4 tiered proxy: `50.50 tok/s`
- exact self-spec tok/s-push proxy: `67.91-68.18 tok/s`

This is still not a real Qwen3-32B inference benchmark. It validates that the
local GPU bandwidth is sufficient for the T002 tok/s proxy if the exact
self-spec path and residency stack are realized.

## T003 Receipt And KV Precision Gate

The first T003 implementation slice adds an opt-in receipt and KV precision
gate:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t003-receipt-kv-gate
```

It emits `T003_RECEIPT_KV_GATE_ACTIVE` and implements:

- A4-20 residency receipt hash;
- A1-02 repair-byte ledger;
- A2-01 KIVI-plus exact newest KV window;
- A6-20 byte-receipt tok/s predictor.

Latest synthetic gate result:

- baseline T002 strong total: `10.739 GiB`
- T003 total without safety: `10.225 GiB`
- T003 total with `0.25 GiB` safety: `10.475 GiB`
- baseline slack: `1.261 GiB`
- T003 slack without safety: `1.775 GiB`
- T003 slack with safety: `1.525 GiB`
- saved vs T002 strong: `0.514 GiB`
- KV residency at 16k with 1024 exact recent tokens and old K/V `2/3` bits:
  `0.866 GiB`
- baseline tok/s proxy: `51.34 tok/s`
- T003 tok/s proxy: `63.66 tok/s`
- proxy speedup: `1.240x`

This is a synthetic receipt and denominator gate, not measured Qwen3-32B VRAM,
quality, or generation throughput.

## T004 Real Qwen Runtime Bridge

The T004 bridge is the first opt-in real-artifact preflight:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t004-real-qwen-bridge
```

It emits `T004_REAL_QWEN_BRIDGE_ACTIVE` and checks:

- Qwen3-32B config shape against the T001-T003 denominator;
- local CUDA, package, VRAM, and disk state;
- local `.safetensors` shard availability;
- header-only safetensors inventory when weights are present;
- concrete blockers before any full model load or generation attempt.

Latest bridge smoke:

- config source: Hugging Face cache snapshot;
- config shape match: `true`;
- local Qwen3-32B safetensors present: `true`;
- safetensors shards: `17`;
- tensor count: `707`;
- tensor payload: `61.024 GiB`;
- blocker: `full_compressed_generation_runtime_not_implemented`;
- generation ready: `false`.

This is not a real Qwen generation benchmark. It is the organized bridge that
prevents the synthetic line from being mistaken for a loaded model runtime.

The raw official BF16 GPU-only smoke is repeatable with:

```powershell
python scripts/research/t004_raw_qwen_cuda_smoke.py --model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --out-dir .work/jobs/t004-raw-qwen-cuda-smoke/latest
```

Latest raw smoke result:

- activation: `T004_RAW_QWEN_CUDA_LOAD_ATTEMPT`;
- result: `OutOfMemoryError` during real parameter materialization;
- free VRAM before load: about `10.78 GiB`;
- raw official GPU-only Qwen3-32B did not reach generation.

This confirms the local weights are usable for inspection, but the project still
needs the compressed residency runtime before a real 12 GB generation claim.

## T005/T006 Real Project-Path Smokes

T005 reads tiny prefixes from selected official Qwen safetensors and verifies
real tensor shape, finiteness, and sample hashes without loading the full model:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t005-real-weight-sentinel --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137
```

Latest T005 result:

- activation: `T005_REAL_WEIGHT_SENTINEL_ACTIVE`
- sampled tensors: `8`
- passing tensors: `8`
- blockers: none

T006 is the first project-owned compressed-residency smoke over real Qwen
weights. It reads selected real blocks, applies the local ternary sparse repair
codec, reconstructs, and runs a matmul error check:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t006-real-residency-block-smoke --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137
```

Latest T006 result:

- activation: `T006_REAL_RESIDENCY_BLOCK_SMOKE_ACTIVE`
- sampled blocks: `5`
- passing blocks: `4`
- average effective bits per weight: `3.0705`
- average Q4 payload savings: `23.2375%`
- average matmul relative RMSE: `0.4185`
- blockers: none

This is real-weight project-path evidence, not an external Ollama result. It is
still not a full Qwen generation benchmark because the layer/full-model
compressed loader and token loop are not implemented yet.

T007 promotes the same idea to a sliced layer: Q/K/V/O attention plus
gate/up/down MLP projections from one real Qwen layer, exact versus compressed
local paths, and composed residual output comparison.

```powershell
python scripts/research/t001_transformer_residency_probe.py --t007-real-layer-smoke --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137
```

Latest T007 layer 0 result:

- activation: `T007_REAL_LAYER_RESIDENCY_SMOKE_ACTIVE`
- effective bits per weight: `3.0063`
- Q4 payload savings: `24.8413%`
- attention relative RMSE: `0.5290`
- MLP relative RMSE: `0.7448`
- composed layer relative RMSE: `0.0253`
- blockers: none

Layer 63 repeat:

- effective bits per weight: `3.0063`
- Q4 payload savings: `24.8413%`
- attention relative RMSE: `0.5583`
- MLP relative RMSE: `0.7414`
- composed layer relative RMSE: `0.0368`
- blockers: none

T008 promotes the same project path into a sliced multi-layer generation loop:
real Qwen embeddings, selected layer projections, and lm_head slices are loaded
from safetensors, compressed through the project codec, and used to generate
tokens autoregressively.

```powershell
python scripts/research/t001_transformer_residency_probe.py --t008-compressed-generation-smoke --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137
```

Latest T008 default result:

- activation: `T008_COMPRESSED_GENERATION_SMOKE_ACTIVE`
- layers: `[0, 1]`
- generated tokens: `[478, 141, 440, 163]`
- exact top-1 tokens: `[141, 103, 440, 490]`
- top-1 agreement: `0.25`
- average logits relative RMSE: `0.4939`
- max logits relative RMSE: `0.5145`
- effective bits per weight: `3.2360`
- Q4 payload savings: `19.0993%`
- sliced compressed tok/s: `865.6884`
- blockers: none

Four-layer repeat with `--generation-layer-indices 0,1,2,3` also passed with
`3.1365` effective bits per weight, `21.5875%` Q4 payload savings, `0.4942`
average logits relative RMSE, `0.50` top-1 agreement, and `634.9710` sliced
compressed tok/s.

This is still sliced micro-generation, not full-token Qwen3-32B generation. It
does not cover full hidden width, all layers, tokenizer-compatible text
generation, KV-cache reuse, GPU kernels, VRAM residency, or real full-model
tok/s.

T009-T013 promote the bridge from sliced generation to full-width projection,
multi-layer scaling, tokenizer, KV-cache pricing, and full-model-equivalent
denominator checks:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t009-t013-runtime-scaler --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137
```

Latest T009-T013 result:

- activation: `T009_T013_QWEN_RUNTIME_SCALER_ACTIVE`
- T009 full-width layer 0 projections: `7/7` pass, `3.0569` effective
  bits/weight, `23.5783%` Q4 payload savings, `0.4367` average projection
  relative RMSE.
- T010 row-capped layers `[0, 1, 2, 3]`: `28/28` projections pass, `3.2427`
  effective bits/weight, `18.9313%` Q4 payload savings, `0.4494` average
  projection relative RMSE.
- T011 tokenizer prompt smoke passes with the local Hugging Face tokenizer.
- T012 KV ledger prices Qwen3-32B FP16 KV at `256 KiB/token`; a 16k tier with
  1024 exact recent tokens and 4-bit old KV is `1.1875 GiB`.
- T013 full-model-equivalent pricing saves `6.3675 GiB` versus Q4 weights plus
  FP16 KV at 16k and projects `43.2359 tok/s` versus `33.6917 tok/s`
  (`1.2833x`), but the current candidate is `13.4382 GiB` with KV and runtime
  overhead, so it does not yet fit the 12 GiB envelope.

This is still not measured full Qwen generation throughput. The projection
timings are CPU/PyTorch eager projection smokes, and T013 is a memory-bound
full-model-equivalent pricing check.

T014 tightens the full-model denominator exposed by T013:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t014-denominator-tightening --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137
```

Latest T014 result:

- activation: `T014_DENOMINATOR_TIGHTENING_ACTIVE`
- T014-A pass: `true`
- MLP target-bit projection gate: full-width layer `0`, `3/3` projections pass,
  `2.4500` effective bits per weight, `38.75%` Q4 payload savings, `0.4848`
  average projection relative RMSE.
- packed arena byte receipt: overhead `0.55 -> 0.25 GiB`, saving `0.3000 GiB`.
- T014-A / F15 projected total: `11.8424 GiB`, `0.1576 GiB` slack,
  `48.5250 tok/s` memory-bound proxy, `1.4403x` versus Q4+FP16KV.
- F21 near-term priced stack: `11.1773 GiB`, `0.8227 GiB` slack,
  `50.5046 tok/s` memory-bound proxy.
- F22 strong priced stack: `10.7187 GiB`, `1.2813 GiB` slack,
  `52.6168 tok/s` memory-bound proxy.

T014 is a real-weight denominator and MLP projection gate. It does not implement
full all-layer Qwen text inference, measured full-model tok/s, or measured 12
GiB GPU residency. The next runtime slice is a packed all-layer compressed
generation path.

T015 adds the first all-layer streamed compressed inference smoke:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t015-packed-qwen-inference --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t015-prompt-text "Local compressed Qwen smoke:" --t015-max-new-tokens 1
```

Latest T015 results:

- activation: `T015_PACKED_QWEN_STREAMED_INFERENCE_ACTIVE`
- layer-1 smoke: pass on CUDA, generated token `89637` / text `HomeAs`,
  `0.1138 tok/s`, peak CUDA allocated `0.4982 GiB`.
- all-64 smoke: pass on CUDA, generated token `220` / text `" "`,
  completed `64/64` layers, `0.0047 tok/s`, peak CUDA allocated `0.4982 GiB`.
- T014-A denominator used by the smoke: `11.8424 GiB`, `0.1576 GiB` slack.

T015 proves functional all-layer token generation through the project
compressed-weight path. It is not optimized packed residency throughput: the
current path streams and reconstructs weights on the fly, so speed is expected
to be terrible until a persistent packed GPU arena or cached project format is
implemented.

T016 adds the pre-kernel persistent packed cache:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t016-persistent-packed-cache --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t016-prompt-text "Local compressed Qwen smoke:" --t016-max-new-tokens 1
```

Latest T016 results:

- activation: `T016_PERSISTENT_PACKED_CACHE_ACTIVE`
- unit suite: `25` tests passed.
- layer-1 cache refresh/build: pass on CUDA, `82` tensors built, generated
  token `89637` / text `HomeAs`, `10.3344s`, `0.096764 tok/s`.
- layer-1 hot cache: pass on CUDA, `82` tensors loaded, same token/text,
  `2.5044s`, `0.399302 tok/s`.
- all-64 cache fill: pass on CUDA, completed `64/64`, built `441`, loaded `82`,
  generated token `220` / text `" "`, `229.9812s`, `0.004348 tok/s`.
- all-64 hot cache: pass on CUDA, completed `64/64`, built `0`, loaded `523`,
  generated token `220` / text `" "`, `43.9065s`, `0.022776 tok/s`.
- full persistent cache: `523` files and `32.8003 GiB` read on the hot all-64
  run.
- peak CUDA allocated/reserved: `1.0080 GiB` / `1.3320 GiB`.
- T014-A denominator remains `11.8424 GiB` total with `0.1576 GiB` slack.

T016 is `4.85x` faster than T015 all-64 streamed inference on this smoke, but
it is still a pre-kernel boundary. The hot run is dominated by `torch.load` and
dense reconstruction (`23.5721s` load, `19.0640s` reconstruction), so the next
runtime slice is fused dequant/matmul directly over packed payloads.

T017 adds the first packed-linear Triton kernel:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t017-kernel-qwen-inference --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --t017-prompt-text "Local compressed Qwen smoke:" --t017-max-new-tokens 1
```

Latest T017 results:

- activation: `T017_PACKED_LINEAR_KERNEL_ACTIVE`
- backend: Triton on PyTorch `2.10.0+cu128`.
- unit suite: `27` tests passed.
- synthetic packed-linear gates versus T016 dense reconstruct+matmul:
  `2.5119x` on `512x512`, `3.8805x` on attention-like `8192x5120`, and
  `4.7723x` on MLP-like `25600x5120`, all with relative RMSE below `1e-6`.
- layer-1 Qwen hot repeat: `1/1`, generated `89637` / `HomeAs`, `1.5295s`,
  `0.653822 tok/s`, peak CUDA allocated `0.1476 GiB`.
- all-64 Qwen hot final: `64/64`, generated `220` / `" "`, `35.6344s`,
  `0.028063 tok/s`, peak CUDA allocated `0.1476 GiB`.
- T017 all-64 is `1.231x` faster than T016 all-64 hot-cache and about `5.97x`
  faster than T015 streamed all-64.

The kernel boundary is now real, but useful tok/s is still blocked by cache I/O:
the all-64 hot final spends `28.0466s` loading `32.8003 GiB` of packed payloads
and only `5.6611s` inside kernels. The next runtime slice should make packed
payloads resident or mmap/binary-loadable before adding more matmul kernels.

T018/T019 close the simple load-path follow-up:

```powershell
python scripts/research/t001_transformer_residency_probe.py --t018-resident-payload-qwen-inference --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --no-hf-config-fetch --t017-layer-limit 1 --t017-max-new-tokens 1 --t017-device cuda --t017-lm-head-chunk-rows 2048
python scripts/research/t001_transformer_residency_probe.py --t019-binary-payload-qwen-inference --local-model-dir C:\Users\th1rt\.cache\huggingface\hub\models--Qwen--Qwen3-32B\snapshots\9216db5781bf21249d130ec9da846c4624c16137 --no-hf-config-fetch --t017-layer-limit 1 --t017-max-new-tokens 1 --t017-device cuda --t017-lm-head-chunk-rows 2048
```

Latest T018/T019 results:

- activations: `T018_RESIDENT_PAYLOAD_RUNTIME_ACTIVE`,
  `T019_BINARY_PAYLOAD_RUNTIME_ACTIVE`.
- unit suite: `27` tests passed.
- same-run T017 layer-1 baseline: `1.5595s`, `0.641224 tok/s`.
- T018 resident CPU payload arena: functionally passes and reproduces token
  `89637` / `HomeAs`, but regresses to `3.5424s`, `0.282297 tok/s`; all-64
  resident CPU payloads were unsafe with only about `16.11 GiB` RAM free versus
  the known `32.8003 GiB` packed-payload footprint.
- T019 binary/mmap sidecars: functionally passes and reproduces the same token,
  but hot binary regresses to `4.3629s`, `0.229205 tok/s`.

Conclusion: simple CPU residency and naive binary mmap are measured-negative.
The next work needs a fresh research round around execution shape, not another
small cache-container patch: likely GPU/pinned staged transfer, payload batching,
true KV decode, or another way to reduce per-token transfer/materialization.

External `qwen3:32b` Ollama generation was tested only as a non-claim sanity
check. It generated visible text, but its `2.34-4.08 tok/s` API samples are
expected to be dominated by Ollama's own quantization/offload path and must not
be reported as this project's compressed-residency throughput.

## Source Of Truth

Research closeout:

- `.workspace/research/t001-qwen3-32b-dense-gpu-residency/extraordinary-synthesis.md`
- `.workspace/research/t002-transformer-kv-margin-efficiency/extraordinary-synthesis.md`
- `.workspace/research/t003-transformer-ram-efficiency-extraordinary/extraordinary-synthesis.md`
- `.workspace/research/t004-real-qwen-runtime-bridge/bridge-brief.md`
- `.workspace/research/t014-full-model-denominator-tightening/synthesis.md`
- `.workspace/research/t015-packed-qwen-inference/implementation-log.md`
- `.workspace/research/t018-t019-load-path/implementation-log.md`

Continuity:

- `.workspace/codex/STATE.md`
- `.workspace/codex/QUEUE.md`

Durable orientation:

- `memory/transformer-residency.md`
