# Baseline Shard 005

## R441 T023 Real Qwen Runtime Implementation Measurement (2026-05-26)

Source: `R441` / `T023`

Artifacts:

- `src/transformer_residency/t023_qwen_runtime.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t023-qwen-runtime/all64-final/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t017-kernel-qwen-inference/all64-same-window/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t022-qwen-runtime/all64-real-rerun/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t022-qwen-runtime/all64-budget4-rerun/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t019-binary-payload-runtime/all64-same-window/summary.json`
- `.workspace/research/t023-qwen-vram-efficiency-extraordinary/real-runtime-implementation-log.md`

Implementation:

The stale T023 cumulative projection gate was removed. The implemented opt-in
runtime path is `--t023-qwen-runtime`, emitting
`T023_QWEN_RUNTIME_ACTIVE`. The default path runs the real T017 all-layer Qwen
token loop with no CUDA payload residency, dense CUDA tensor cache, packed work
coalescing, device credit-flow receipts, and on-demand register-decoded packed
linear kernels.

Same-window low-VRAM comparison:

| path | elapsed seconds | tokens/s | peak CUDA allocated GiB | token |
|---|---:|---:|---:|---:|
| T017 same-window baseline | `52.26120319997426` | `0.01913465321824991` | `0.14755725860595703` | `220` |
| T023 real runtime default | `50.82945140002994` | `0.019673633542293376` | `0.1500396728515625` | `220` |

Derived low-VRAM comparison:

- T023 / T017 tokens/s ratio: `1.028166x`.
- Elapsed saved: `1.4317517999457195s`.
- Kernel seconds changed from `7.10984870075481s` to
  `5.128353500214871s`.
- Peak CUDA allocation increased by about `0.0024824142456054688 GiB`
  (`2.54 MiB`).

Additional same-window context:

| path | elapsed seconds | tokens/s | peak CUDA allocated GiB | status |
|---|---:|---:|---:|---|
| T022 9.5 GiB device payload residency | `25.099291599995922` | `0.03984176190853779` | `9.747875213623047` | faster but high VRAM |
| T022 4.0 GiB bounded residency first run | `29.757666099991184` | `0.03360478596136598` | `4.192046642303467` | promising but not stable |
| T023 4.0 GiB bounded residency repeat | `50.246817200037185` | `0.01990175807591769` | `4.192046642303467` | not stable |
| T019 binary payload path | `75.52321439998923` | `0.013240961841265891` | `0.14755725860595703` | worse |

Status:
`PASS_T023_REAL_RUNTIME_IMPLEMENTED_INSUFFICIENT_SPEEDUP`.

Claim boundary:
This is a real all-layer Qwen runtime measurement through the project
compressed-weight path, not a projection gate and not a synthetic proof object.
It does not satisfy the `>=40 tok/s` project target. The current measured
blocker is hot-path payload I/O / `torch.load`; the 4 GiB residency point is not
stable enough to promote, and the binary sidecar path regressed in the measured
all-layer run.

---

## R442 T024 GPU-Bound Low-VRAM Runtime Implementation Measurement (2026-05-26)

Source: `R442` / `T024`

Artifacts:

- `src/transformer_residency/t024_qwen_runtime.py`
- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/runtime_memory.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t023-qwen-runtime/t24-compare/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t024-qwen-runtime/t24-compare/summary.json`
- `.workspace/research/t024-gpu-bound-vram-efficiency-extraordinary/implementation-result.md`

Implementation:

T024 is an opt-in real Qwen runtime path exposed as `--t024-qwen-runtime`,
emitting `T024_QWEN_RUNTIME_ACTIVE`. It keeps the T023 GPU-bound low-VRAM
policy: no CUDA payload residency, no payload preload, dense CUDA tensor cache,
packed work coalescing, and device credit-flow receipts. The landed T024
runtime change avoids materializing a third MLP intermediate tensor by applying
the gate/up product in place before the down projection. The runtime receipt now
records process RAM metrics through Windows PSAPI.

Same-window T23/T24 comparison:

| path | elapsed seconds | tokens/s | peak CUDA allocated GiB | peak CUDA reserved GiB | peak process working set GiB | process private usage GiB | token |
|---|---:|---:|---:|---:|---:|---:|---:|
| T23 real runtime default | `32.52012429997558` | `0.03075018996777792` | `0.1500396728515625` | `0.20703125` | `1.55792236328125` | `2.9703750610351562` | `220` |
| T24 in-place MLP runtime | `32.44632749998709` | `0.030820129026941426` | `0.1500396728515625` | `0.20703125` | `1.5573997497558594` | `2.990955352783203` | `220` |

Derived comparison:

- T24 / T23 tokens/s ratio: `1.002274x`.
- Elapsed saved: `0.07379679998848587s`.
- Kernel seconds changed from `5.09177819971228s` to `5.036097300355323s`.
- Peak CUDA allocated did not improve: delta `0.0 GiB`.
- Peak CUDA reserved did not improve: delta `0.0 GiB`.
- Peak process working set changed by `-0.000522613525390625 GiB`.
- Process private usage changed by `+0.020580291748046875 GiB`.

Status:
`PASS_T024_REAL_RUNTIME_IMPLEMENTED_NO_GPU_MEMORY_GAIN`.

Claim boundary:
This is a real all-layer Qwen runtime measurement through the compressed-weight
path, not a projection. T024 is implemented and functional, but this landed
piece did not reduce allocator-visible GPU peak memory in the one-token all-layer
run.

---

## R443 T024 Hot Trit-Packed Runtime Measurement (2026-05-26)

Source: `R443` / `T024`

Artifacts:

- `src/transformer_residency/t016_persistent_packed_cache.py`
- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/t024_qwen_runtime.py`
- `src/transformer_residency/t024_exact_quotients.py`
- `.work/jobs/t001-transformer-residency-probe/t023-qwen-runtime/t24-trit-compare/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t024-qwen-runtime/t24-trit-cache-build/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t024-qwen-runtime/t24-trit-hot-compare/summary.json`
- `.workspace/research/t024-gpu-bound-vram-efficiency-extraordinary/implementation-result.md`

Implementation:

T024 now adds radix-3 trit-packed ternary code pages decoded directly inside
the Triton packed-linear kernels, plus a persistent `trit-payloads/` cache so
hot runs do not repack codes on CPU. T024 still keeps the GPU-bound low-VRAM
policy: no CUDA payload residency, no payload preload, dense CUDA scalar/vector
cache, packed work coalescing, in-place MLP gate/up product, and process RAM
receipts. Metadata exact-quotient checks found no lm-head/embedding alias and no
invalid lm-head rows for this Qwen artifact.

Same-window hot comparison:

| path | elapsed seconds | tokens/s | peak CUDA allocated GiB | peak CUDA reserved GiB | peak process working set GiB | process private usage GiB | token |
|---|---:|---:|---:|---:|---:|---:|---:|
| T23 real runtime default | `39.29277479997836` | `0.025449971530148864` | `0.1500396728515625` | `0.20703125` | `1.5957527160644531` | `2.9821510314941406` | `220` |
| T24 hot trit-packed runtime | `11.002756200032309` | `0.0908863181024645` | `0.0582427978515625` | `0.0859375` | `1.4516334533691406` | `2.7767562866210938` | `220` |

Derived comparison:

- T24 / T23 tokens/s ratio: `3.570198x`.
- Elapsed saved: `28.290018599946052s`.
- Kernel seconds changed from `7.431253200455103s` to
  `2.4220448008854873s`.
- Peak CUDA allocated reduced by `0.091796875 GiB`.
- Peak CUDA reserved reduced by `0.12109375 GiB`.
- Peak process working set reduced by `0.1441192626953125 GiB`.
- Process private usage reduced by `0.20539474487304688 GiB`.
- T24 hot run loaded `523` trit payloads, read `10.460136723704636 GiB` from
  the trit cache, and avoided `29.7869873046875 GiB` of unpacked code bytes.

Status:
`PASS_T024_HOT_TRIT_RUNTIME_IMPLEMENTED_REAL_GPU_MEMORY_AND_SPEED_GAIN`.

Claim boundary:
This is a real all-layer Qwen runtime measurement through the compressed-weight
path. It depends on a built trit payload cache; the first cache-build run is not
the hot-runtime number.

---

## R444 T025 Fail-Closed GPU Resident Trit Runtime Measurement (2026-05-26)

Source: `R444` / `T025`

Artifacts:

- `src/transformer_residency/t016_persistent_packed_cache.py`
- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/t025_qwen_runtime.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t024-qwen-runtime/t025-compare-layer1/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t025-qwen-runtime/layer1-compact-dtypes-hot/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t025-qwen-runtime/all64-compact-dtypes/summary.json`

Implementation:

T025 adds a fail-closed CUDA-resident compact payload arena on top of the T024
trit-packed runtime. Packed trit payloads, scales, repair values, and repair
indices are preloaded into CUDA memory before the measured token loop. The hot
path consumes a device payload descriptor table and aborts on any payload miss
instead of falling back to `torch.load` or CPU/RAM staging. The resident payload
keeps compact dtypes for scales, repair values, and repair indices, converting
to fp32 only inside the Triton accumulator.

Same-window layer-1 hot comparison:

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | peak CUDA reserved GiB | device resident GiB | hot payload load events | hot payload load seconds | token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T24 layer-1 hot trit runtime | `0.7848702999763191` | `1.27409586020795` | `0.2991733996896073` | `0.055779457092285156` | `0.083984375` | `0.0` | `82` | `0.18226210039574653` | `89637` |
| T25 layer-1 hot resident runtime | `0.496893799980171` | `2.012502470427093` | `0.21736670011887327` | `0.5868539810180664` | `0.609375` | `0.570422887802124` | `0` | `0.0` | `89637` |

Derived layer-1 comparison:

- T25 / T24 tokens/s ratio: `1.579555x`.
- Elapsed saved: `0.28797649999614805s`.
- Kernel seconds reduced by `0.08180669957073403s`.
- Peak CUDA allocated increased by `0.5310745239257812 GiB`, intentionally,
  because T25 makes compact payload residency real.
- Hot payload load events changed from `82` to `0`.

All-layer hot comparison against R443:

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | peak CUDA reserved GiB | device resident GiB | hot payload load events | hot payload load seconds | token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T24 all-layer hot trit runtime | `11.002756200032309` | `0.0908863181024645` | `2.4220448008854873` | `0.0582427978515625` | `0.0859375` | `0.0` | `523` | `7.8987952009774745` | `220` |
| T25 all-layer resident runtime | `1.35226660000626` | `0.7394991490549059` | `0.8200247992645018` | `10.659311294555664` | `10.724609375` | `10.458633184432983` | `0` | `0.0` | `220` |

Derived all-layer comparison:

- T25 / T24 tokens/s ratio: `8.136521x`.
- Elapsed saved: `9.650489600026049s`.
- Kernel seconds reduced by `1.6020200016209855s`.
- Peak CUDA allocated increased by `10.601068496704102 GiB`, intentionally,
  because T25 is now GPU-resident instead of fake-low-VRAM streaming.
- T25 resident descriptor count: `523`.
- T25 compact resident payload: `10.458633184432983 GiB`.
- T25 hot payload load events: `0`.
- T25 hot payload load seconds: `0.0`.

Status:
`PASS_T025_FAIL_CLOSED_GPU_RESIDENT_RUNTIME_IMPLEMENTED_MAJOR_SPEEDUP_STILL_KERNEL_BOUND`.

Claim boundary:
This is a real all-layer Qwen runtime measurement through the project
compressed-weight path. It is no longer a low-VRAM streaming path: it
intentionally uses most of the RTX 5070 VRAM to keep compact payloads resident
and remove hot `torch.load`. It still does not satisfy the `>=40 tok/s` project
target. After T025, the dominant blocker is native compact kernel and layer
fusion efficiency, not payload load.

## R445 T026 GPU-Owned Decode Orchestration Measurement (2026-05-26)

Source: `R445` / `T026`

Artifacts:

- `src/transformer_residency/t026_qwen_runtime.py`
- `src/transformer_residency/t017_packed_linear_kernel.py`
- `.work/jobs/t001-transformer-residency-probe/t026-qwen-runtime/all64-deferred-sync-final/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t025-qwen-runtime/all64-same-window-after-t026/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t026-qwen-runtime/all64-fused-mlp-gate-up/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t026-qwen-runtime/all64-block-m32/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t026-qwen-runtime/all64-block-c256/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t026-qwen-runtime/all64-block-c64/summary.json`

T026 keeps the T025 fail-closed CUDA-resident compact payload contract and adds
GPU-owned orchestration that defers per-projection CUDA synchronization. It also
implements opt-in probes for device-side `lm_head` top-1 accumulation, larger
Triton compact-affine tiles, and a fused MLP `gate_proj + up_proj` pair kernel;
those probes were measured and rejected as default paths because they regressed
all-layer tok/s.

All rows generated token `220` with full `64/64` layers, `523` descriptors,
`0` hot payload load events, and `0.0` hot payload load seconds.

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | default promoted? |
|---|---:|---:|---:|---:|---:|---|
| T25 same-window resident runtime | `1.6618152000010014` | `0.6017516267749853` | `1.0036989009822719` | `10.659311294555664` | `10.458633184432983` | previous default |
| T26 deferred-sync runtime | `1.156065299990587` | `0.8650030409252335` | `0.4827828996349126` | `10.659311294555664` | `10.458633184432983` | yes |
| T26 GPU `lm_head` top-1 accumulator | `3.8998610000126064` | `0.256419395459676` | `2.9476224997779354` | `10.659311294555664` | `10.458633184432983` | no |
| T26 larger `lm_head` chunks, 8192 rows | `1.756918200000655` | `0.5691784626055028` | `0.8930393004557118` | `10.658791542053223` | `10.458633184432983` | no |
| T26 `BLOCK_M=32` compact affine | `4.079794200020842` | `0.24511040287152017` | `3.3036541997571476` | `10.659311294555664` | `10.458633184432983` | no |
| T26 `BLOCK_C=256` compact affine | `8.806761800020467` | `0.11354911404526416` | `5.186319500789978` | `10.659311294555664` | `10.458633184432983` | no |
| T26 `BLOCK_C=64` compact affine | `3.091607499984093` | `0.3234563249070735` | `2.4185798001126386` | `10.659311294555664` | `10.458633184432983` | no |
| T26 fused MLP gate/up pair kernel | `4.421502100012731` | `0.22616748276499082` | `3.774800400075037` | `10.659883499145508` | `10.458633184432983` | no |

Comparisons:

- T26 deferred-sync versus R444 T25: `1.1697147211470411x` tok/s.
- T26 deferred-sync versus same-window T25: `1.4374752014566412x` tok/s.
- T26 keeps the T25 fail-closed residency invariants: no hot payload movement,
  no CPU fallback, and the same generated token.

Status:
`PASS_T026_DEFERRED_SYNC_GPU_OWNED_RUNTIME_IMPLEMENTED_WITH_REJECTED_FUSION_PROBES`.

Claim boundary:

T026 is a real all-layer Qwen runtime improvement, not a synthetic gate. It is
still nowhere near the `>=40 tok/s` target. The implemented simple MLP pair
fusion and tile retunings are not the needed native compact kernel; they
increase register/launch pressure and regress. The next honest frontier requires
a redesigned fused compact layer kernel rather than another wrapper around the
current per-projection Triton decode kernel.

---

## R446 T027 Cumulative Generation-Speed Stack Measurement (2026-05-26)

Source: `R446` / `T027`

Artifacts:

- `src/transformer_residency/t027_qwen_runtime.py`
- `src/transformer_residency/t017_packed_linear_kernel.py`
- `.work/jobs/t001-transformer-residency-probe/t027-qwen-runtime/latest/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t027-qwen-runtime/repeat-01/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t026-qwen-runtime/same-window-t027/summary.json`
- `.workspace/research/t027-generation-speed-extraordinary/synthesis.md`

T027 implements `--t027-qwen-runtime` as a cumulative generation-speed stack on
top of T026. It keeps the T025/T026 fail-closed CUDA-resident compact payload
contract and activates the existing exact runtime mechanisms corresponding to
the T027 validated findings: trit packed compact-affine codes, descriptor table,
deferred CUDA synchronization, dense CUDA scalar/vector cache, device credit-flow
receipt, packed work coalescing, in-place MLP intermediate, fused MLP gate/up
pair kernel, and GPU greedy `lm_head` tournament. It also records the validated
T027 continuous-generation projection envelope, while marking that the projection
does not robustly cross `40 tok/s`.

All rows generated token `220` with full `64/64` layers, `523` descriptors,
`0` hot payload load events, `0.0` hot payload load seconds, and
`10.458633184432983 GiB` compact payload resident on CUDA.

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | notes |
|---|---:|---:|---:|---:|---:|---|
| T026 same-window baseline | `1.432813600054942` | `0.6979274903320672` | `0.6154180009034462` | `10.659883499145508` | `10.458633184432983` | T026 deferred-sync stack |
| T027 first all-layer run | `1.1354936000425369` | `0.8806742723715386` | `0.4143302993616089` | `10.659883499145508` | `10.458633184432983` | full cumulative stack active |
| T027 repeat all-layer run | `1.0904432000243105` | `0.9170583116825396` | `0.528349399624858` | `10.659883499145508` | `10.458633184432983` | full cumulative stack active |

Comparisons:

- T027 first run versus same-window T026: `1.2618420746724309x` tok/s.
- T027 repeat run versus same-window T026: `1.3139736210221667x` tok/s.
- T027 first run versus R445 T026: `1.0181169668831946x` tok/s.
- T027 repeat run versus R445 T026: `1.0601792922041362x` tok/s.
- T027 repeat remains `43.61661652158832x` below the `40 tok/s` target.

Status:
`PASS_T027_CUMULATIVE_GENERATION_SPEED_STACK_IMPLEMENTED_INSUFFICIENT_SPEED`.

Claim boundary:

T027 is a real all-layer Qwen runtime path with all current exact cumulative
mechanisms activated. It is not a robust generation-speed breakthrough. The
activated wrapper-level and per-projection mechanisms improve same-window T026
by about `1.26x-1.31x`, but measured throughput remains below `1 tok/s` and far
below the `>=40 tok/s` target. The validated T027 research projection still
requires a real native whole-token compact layer/supernode kernel and high-density
A03 continuous greedy certificates; the current implementation does not provide
those missing native kernels.

---

## R447 T028 Native Generation-Speed Substrate Measurement (2026-05-26)

Source: `R447` / `T028`

Artifacts:

- `src/transformer_residency/t028_native_generation_speed.py`
- `tests/test_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t028-native-generation-speed/all64-actual/summary.json`
- `.workspace/research/t028-native-generation-speed-extraordinary/implementation-log.md`

T028 implements `--t028-native-generation-speed` as a cumulative native
generation-speed substrate stack on top of T027. It keeps the T025/T026/T027
CUDA-resident compact payload path and adds executable receipts for the T028
shortlist: fixed-address CUDA graph capture, device argmax-to-embedding closure,
stable virtual KV cursor/page-boundary update, four-layer macro-phase graph,
static token trace, multi-token device chunk, persistent token-service state,
single-request gate/up packetization, persistent reduction fabric, compact
affine Triton tile microgate, streaming MLP contraction, attention streaming
boundary, and an offline certificate-density gate.

All rows generated token `220` with full `64/64` layers, `523` descriptors,
`0` hot payload load events, `0.0` hot payload load seconds, and
`10.458633184432983 GiB` compact payload resident on CUDA.

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | notes |
|---|---:|---:|---:|---:|---:|---|
| R446 T027 repeat | `1.0904432000243105` | `0.9170583116825396` | `0.528349399624858` | `10.659883499145508` | `10.458633184432983` | latest relevant baseline |
| T028 all-layer substrate stack | `1.4639156000339426` | `0.683099490146026` | `0.6680870002019219` | `10.659883499145508` | `10.458633184432983` | sidecar native substrate gates active |

Implemented T028 gate receipts:

- fixed-address CUDA graph captured and replayed with `0.0` max replay diff;
- device argmax-to-embedding closure proved;
- virtual KV cursor crossed a page boundary without host cursor update;
- four-layer macro-phase graph captured and matched eager output with `0.0`
  max diff;
- static token trace emitted `523` descriptors;
- multi-token device chunk produced device sequence `[7, 15, 24, 34]`;
- compact affine tile microgate passed at `1.8542516437349879x` versus
  reconstruct+matmul with `2.7752369646805164e-07` relative RMSE;
- streaming MLP contraction removed `1280` hidden bytes in the microgate and
  matched within `1.1920928955078125e-07`;
- attention streaming boundary matched within `2.3283064365386963e-09`;
- certificate-density gate accepted `0.796875` of samples with no false skip.

Comparison:

- T028 / R446 T027 repeat tokens/s ratio: `0.7448811939698076x`.
- T028 is `58.55662400135771x` below the `40 tok/s` target.
- Peak CUDA allocation and compact resident payload were unchanged.

Status:
`PASS_T028_NATIVE_SUBSTRATE_RECEIPTS_IMPLEMENTED_ALL_LAYER_SPEED_REGRESSION`.

Claim boundary:

T028 now implements the requested native substrate receipts, but they are still
sidecar proofs around the current T027 per-projection Python/Triton token loop.
They do not yet replace the real hot loop with a graph-captured whole-token
program or fused native layer/supernode kernels. The all-layer measurement is a
regression, so no production tok/s speedup claim is allowed.

---

## R452 T028 Hot-Loop Partial Integration Measurement (2026-05-26)

Source: `R452` / `T028`

Artifacts:

- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/t027_qwen_runtime.py`
- `src/transformer_residency/t028_native_generation_speed.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t028-native-generation-speed/r452-t028-default-hotloop/summary.json`

This update moves part of T028 out of sidecar-only receipts and into the real
all-layer hot loop:

- prompt embedding rows are preloaded on device before the measured token loop;
- causal attention uses the real `scaled_dot_product_attention` boundary;
- T028 gets its own compact-affine tile default, `8x128x32`, without changing
  the T027 default.

All rows generated token `220` with full `64/64` layers, `523` descriptors,
`0` hot payload load events, `0.0` hot payload load seconds, and
`10.458633184432983 GiB` compact payload resident on CUDA.

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | notes |
|---|---:|---:|---:|---:|---:|---|
| R446 T027 repeat | `1.0904432000243105` | `0.9170583116825396` | `0.528349399624858` | `10.659883499145508` | `10.458633184432983` | latest relevant baseline |
| R447 T028 sidecar receipts | `1.4639156000339426` | `0.683099490146026` | `0.6680870002019219` | `10.659883499145508` | `10.458633184432983` | sidecar receipts only |
| R452 T028 hot-loop partial integration | `1.0775478000286967` | `0.9280330765589875` | `0.5049169003032148` | `10.652063369750977` | `10.458633184432983` | prompt embedding preload, SDPA attention, T028 tile `8x128x32` |

Comparison:

- R452 / R446 T027 repeat tokens/s ratio: `1.0119673577313881x`.
- R452 is still `43.101912001147866x` below the `40 tok/s` target.
- The measured gain is below the project `10%` materiality floor.

Status:
`PASS_T028_HOT_LOOP_PARTIAL_INTEGRATION_BELOW_MATERIALITY`.

Claim boundary:

T028 now has real hot-loop integration for prompt embedding preload, SDPA
attention, and a T028-specific compact-affine tile default. This is still not a
material production speedup. The remaining denominator requires replacing the
Python per-layer/per-projection loop with CUDA graph or native macro-phase /
supernode execution, or replacing the compact-affine inner loop itself.

---

## R455 T029 Exact 20-Trit-Equivalent GPU Payload Measurement (2026-05-26)

Source: `R455` / `T029`

Artifacts:

- `src/transformer_residency/t016_persistent_packed_cache.py`
- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/t029_speed_gpu_efficiency.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.work/jobs/t001-transformer-residency-probe/t029-speed-gpu-efficiency/r454-t028-current-control/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t029-speed-gpu-efficiency/r455-t029-exact20-lanes5/summary.json`

T029 implements `--t029-speed-gpu-efficiency` as the T028 hot-loop path plus an
exact 20-trit-equivalent base-3 payload recode. The active implementation stores
the 20-trit word as four exact 5-trit byte lanes, preserving the same
`1.6 bits/trit` code density while avoiding the slower 32-bit base-3 division
path measured in `R453`.

Same-window one-layer control:

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | process working set GiB | process private GiB | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R454 T028 current control, 4-trit bytes | `2.5681074999738485` | `0.3893917992179779` | `2.4080306001123972` | `0.5796060562133789` | `0.570422887802124` | `1.2531013488769531` | `3.0656204223632812` | `82` descriptors, `0` hot payload loads |
| R455 T029 20-trit-equivalent lanes | `2.7298531999695115` | `0.36632006439436693` | `2.5700341000338085` | `0.5299234390258789` | `0.5114963054656982` | `3.935516357421875` | `5.729991912841797` | `82` descriptors, `0` hot payload loads |

Comparison:

- Resident CUDA payload improved by `0.05892658233642578 GiB`
  (`0.570422887802124 -> 0.5114963054656982 GiB`, `10.329%` less).
- Peak CUDA allocation improved by `0.0496826171875 GiB`
  (`0.5796060562133789 -> 0.5299234390258789 GiB`, `8.572%` less).
- Tokens/s regressed from `0.3893917992179779` to `0.36632006439436693`
  (`0.940746x`, about `5.925%` slower).
- Process working set and private usage increased in the candidate run; this is
  not a host-RAM efficiency win.

Status:
`PASS_T029_EXACT_20_TRIT_EQUIVALENT_GPU_PAYLOAD_MEMORY_WIN_SPEED_REGRESSION`.

Claim boundary:

T029 is a real GPU-resident payload implementation and it reduces CUDA-resident
payload bytes without CPU offload. It is not a speed win. The 20-trit-equivalent
decode still costs enough in the current Triton compact-affine kernel to regress
tok/s, so the codec should not be promoted as the default generation path until
a faster decode kernel or another no-regression memory path is implemented.

---

## R456 T030 Cumulative Speed And GPU Efficiency One-Layer Smoke (2026-05-26)

Source: `R456` / `T030`

Artifacts:

- `src/transformer_residency/t030_speed_gpu_efficiency.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.workspace/research/t030-speed-gpu-efficiency-extraordinary/implementation-log.md`
- `.work/jobs/t001-transformer-residency-probe/t030-speed-gpu-efficiency/one-layer-t029-control/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t030-speed-gpu-efficiency/one-layer-smoke-rerun/summary.json`

T030 implements `--t030-speed-gpu-efficiency` as a cumulative opt-in path over
T029/T028/T027/T025. It keeps the exact 5-trit lane payload, fail-closed
CUDA-resident payload arena, native SDPA attention boundary, MLP gate/up pair
kernel, MLP down residual coalescing, device descriptor table, and GPU argmax
head. It also records metadata-only gates for exact tile quotient, zero/skew
superblocks, and cross-layer residual basis coverage. Those metadata gates are
not direct structural memory-savings claims.

Same-window one-layer control:

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | process working set GiB | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| T029 control, 5-trit lanes | `0.3939284000080079` | `2.5385323829905935` | `0.22053609957220033` | `0.5299234390258789` | `0.5114963054656982` | `3.893474578857422` | token `89637`, `0` hot payload loads |
| T030 cumulative candidate | `0.32442610000725836` | `3.0823660611079906` | `0.1579949003062211` | `0.5299234390258789` | `0.5114963054656982` | `3.894847869873047` | token `89637`, `0` hot payload loads |

Comparison:

- T030/T029 same-window one-layer tokens/s ratio:
  `1.214229969551859x`.
- Kernel seconds improved from `0.22053609957220033` to
  `0.1579949003062211`.
- Peak CUDA and CUDA-resident payload memory were unchanged in this same-window
  one-layer control.
- Process working set was effectively unchanged at this scale
  (`3.893474578857422 -> 3.894847869873047 GiB`); no host-RAM win is claimed.

Status:
`PASS_T030_CUMULATIVE_STACK_ONE_LAYER_SMOKE_NON_COMPARABLE_TO_R452`.

Claim boundary:

This is a real opt-in T030 runtime path and it passed a one-layer smoke. It is
not comparable to the R452/T028 full all-layer production baseline. No
production tok/s claim, R452 speedup claim, or 40 tok/s claim is allowed until a
clean-window full `64/64` all-layer T030 run completes without CPU/RAM offload.

---

## R457 T030 QKV Triple Compact Projection Kernel Smoke (2026-05-26)

Source: `R457` / `T030`

Artifacts:

- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/t030_speed_gpu_efficiency.py`
- `.workspace/research/t030-speed-gpu-efficiency-extraordinary/implementation-log.md`
- `.work/jobs/t001-transformer-residency-probe/t030-speed-gpu-efficiency/one-layer-qkv-triple/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t030-speed-gpu-efficiency/one-layer-qkv-triple-repeat/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t030-speed-gpu-efficiency/one-layer-no-qkv-control-repeat/summary.json`

R457 implements `qkv_single_compact_projection_kernel` as a real Triton triple
projection kernel. It consumes CUDA-resident trit payloads for Q, K, and V in
one launch and supports different output row counts for Q versus K/V. At R457
time it was enabled by default in `--t030-speed-gpu-efficiency`; R458 reversed
that promotion after the full all-layer validation regressed. The current CLI
uses `--t030-enable-fused-qkv-kernel` to opt into this kernel explicitly.

Same-window one-layer comparison:

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | hot payload loads | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| QKV triple first run | `2.8501033999491483` | `0.3508644633797644` | `2.6799073997535743` | `0.5299234390258789` | `0.5114963054656982` | `0` | first-use Triton compile/launch warmup included |
| QKV triple repeat | `0.35691830003634095` | `2.8017616353607573` | `0.18933859991375357` | `0.5299234390258789` | `0.5114963054656982` | `0` | hot repeat |
| No-QKV repeat control | `0.3751417000312358` | `2.6656594026116958` | `0.17859439965104684` | `0.5299234390258789` | `0.5114963054656982` | `0` | hot repeat with `--t030-disable-fused-qkv-kernel` |

Comparison:

- Hot repeat T030 QKV/no-QKV tokens/s ratio: `1.051808937604461x`.
- CUDA peak and CUDA-resident payload were unchanged.
- Generated token matched: `89637`.
- The first QKV run is not used as a speed claim because it includes first-use
  kernel warmup behavior.

Status:
`PASS_T030_QKV_TRIPLE_KERNEL_ONE_LAYER_SMALL_SPEED_WIN_NON_COMPARABLE_TO_R452`.

Claim boundary:

The QKV item is now a real GPU hot-loop implementation, not a metadata gate. The
gain is small and only measured on one layer. No production all-layer speed
claim, R452 speedup claim, or 40 tok/s claim is allowed until a clean full
`64/64` run validates the cumulative path.

---

## R458 T030 Full All-Layer GPU Validation And QKV Demotion (2026-05-26)

Source: `R458` / `T030`

Artifacts:

- `src/transformer_residency/t030_speed_gpu_efficiency.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.workspace/research/t030-speed-gpu-efficiency-extraordinary/implementation-log.md`
- `.work/jobs/t001-transformer-residency-probe/t030-speed-gpu-efficiency/full64-clean-qkv/summary.json`
- `.work/jobs/t001-transformer-residency-probe/t030-speed-gpu-efficiency/full64-clean-no-qkv/summary.json`

R458 validates T030 on the clean full `64/64` all-layer Qwen path. Both runs
emitted `T030_SPEED_GPU_EFFICIENCY_ACTIVE`, generated token `220`, completed
`64/64` layers, used `523` device payload descriptors, and had `0` hot payload
load events. Both remained GPU-bound with no CPU/RAM offload claim: the CUDA
payload arena stayed resident and fail-closed.

Full all-layer comparison against R452/T028:

| path | elapsed seconds | tokens/s | kernel seconds | peak CUDA allocated GiB | device resident GiB | process working set GiB | private usage GiB | preload seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R452 T028 reference | `1.0775478000286967` | `0.9280330765589875` | `0.5049169003032148` | `10.652063369750977` | `10.458633184432983` | not recorded here | not recorded here | not charged here |
| T030 full no-QKV | `1.8495114999823272` | `0.5406833101657142` | `0.6052840992342681` | `9.255645751953125` | `8.969466924667358` | `3.9428443908691406` | `14.620426177978516` | `79.86799709999468` |
| T030 full QKV | `4.260215000016615` | `0.2347299373379278` | `1.2840290004969575` | `9.255645751953125` | `8.969466924667358` | `3.973857879638672` | `14.621467590332031` | `268.6561129000038` |

Measured deltas:

- T030 no-QKV versus R452/T028 tok/s ratio:
  `0.5826121113812989x`.
- T030 QKV versus R452/T028 tok/s ratio:
  `0.2529327275793579x`.
- T030 QKV versus T030 no-QKV tok/s ratio:
  `0.4341357368350142x`.
- T030 saved `1.3964176177978516 GiB` peak CUDA allocation versus R452/T028
  (`10.652063369750977 -> 9.255645751953125 GiB`).
- T030 saved `1.489166259765625 GiB` CUDA-resident compact payload versus
  R452/T028 (`10.458633184432983 -> 8.969466924667358 GiB`).
- T030 did not improve host RAM; process private usage was about
  `14.62 GiB` in both T030 full runs.

Status:
`PASS_T030_FULL64_GPU_RESIDENT_VRAM_WIN_FAIL_SPEED_REGRESSION_QKV_DEMOTED`.

Claim boundary:

T030 is a real GPU-resident VRAM reduction path and the memory win is above
`1 GiB` without CPU/RAM offload. It is not a speed win and cannot support a
production tok/s, R452 speedup, or `40 tok/s` claim. The QKV triple kernel is
real code, but full all-layer validation makes it explicitly opt-in rather than
default-promoted. Real all-layer CUDA graph capture is still not integrated into
the production hot loop; the existing T028/T030 graph receipts remain sidecar
contracts until a static-buffer graph-captured layer/token program lands.

---

## R459 T033 Runtime Architecture Closure One-Layer Smoke (2026-05-27)

Source: `R459` / `T033`

Artifacts:

- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/t031_macro_architecture_stack.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.workspace/research/t033-real-runtime-architecture-closure/synthesis.md`
- `.work/jobs/t001-transformer-residency-probe/t033-real-runtime-architecture-closure/one-layer-smoke/summary.json`

R459 validates the T033 implementation closeout on the same one-layer Qwen smoke
shape used for the T031/T032 closure line. The CLI emitted
`T031_MACRO_ARCHITECTURE_STACK_ACTIVE`, completed `1/1` layer, generated token
`89637`, and kept the T031 production-claim boundary: this is a
`non_comparable_layer_limited_smoke`, not a full all-layer speed claim.

Measured one-layer runtime:

| path | elapsed scope | tokens/s | peak CUDA allocated GiB | resident CUDA payload GiB | process working set GiB | private usage GiB |
|---|---|---:|---:|---:|---:|---:|
| R458/T032 one-layer reference | layer-limited smoke | `3.223002439344781` | `0.5345873832702637` | `0.5114963054656982` | `4.046268463134766` | `5.84588623046875` |
| R459/T033 one-layer closeout | layer-limited smoke | `2.884471992687007` | `0.5345873832702637` | `0.5114963054656982` | `4.052974700927734` | `5.867942810058594` |

T033 gates:

- `T033-A-V01` greedy LM-head output contract is active and
  implementation-ready. It reports `full_logits_materialized=false`,
  `full_logits_returned_to_host=false`, and `quality_tradeoff=false`.
- The T033 descriptor-denominator gate consumed `82` real device-payload
  descriptors and `549214976` descriptor bytes, but found
  `positive_gate_count=0`.
- Weighted zero-group fraction was only `0.020235216080163126`.
- Repair-free fraction was `0.0`.
- Scale-chamber denominator pricing was negative
  (`447232` saved-work bytes versus `571706` overhead bytes).

Status:
`PASS_T033_GREEDY_CONTRACT_IMPLEMENTED_STRUCTURAL_GATES_NEGATIVE`.

Claim boundary:

T033 implemented the one direct validated item and the structural denominator
gate. It did not improve tok/s, peak CUDA, resident CUDA payload, or host RAM in
this one-layer smoke. The structural payload findings remain blocked until a
new format or kernel substrate changes the denominator; this run explicitly
prevents claiming metadata as a GPU-bound speed or VRAM win.

---

## R460 T034 GPU-Real Runtime Gates One-Layer Smoke (2026-05-27)

Source: `R460` / `T034`

Artifacts:

- `src/transformer_residency/t017_packed_linear_kernel.py`
- `src/transformer_residency/t034_gpu_real_runtime_gates.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.workspace/research/t034-gpu-real-runtime-breakthrough/synthesis.md`
- `.workspace/research/t034-gpu-real-runtime-breakthrough/implementation-log.md`
- `.work/jobs/t001-transformer-residency-probe/t034-gpu-real-runtime-gates/r460-one-layer/summary.json`

R460 implements all three immediate T034 GPU-real gates as opt-in code:

- `T034-A-VAL01` structural execution view over real layer-0 `q_proj`, comparing
  current 5-trit compact-affine execution with a non-dense unpacked-code
  execution view.
- `T034-B-V01/V02` exact GPU segment transaction over pending pages, device
  status word, and serial-state hash equality.
- `T034-C-V01` device argmax-to-next-embedding splice in the real
  `T017` Qwen token loop, with the next embedding gathered on CUDA before the
  host token id is read.

Measured one-layer comparison:

| path | elapsed scope | tokens/s | peak CUDA allocated GiB | resident CUDA payload GiB |
|---|---|---:|---:|---:|
| R459/T033 one-layer reference | layer-limited smoke | `2.884471992687007` | `0.5345873832702637` | `0.5114963054656982` |
| R460/T034 one-layer gates | layer-limited smoke | `0.3735925228602779` | `3.431270122528076` | `0.5114963054656982` |

T034 gate details:

- Structural execution view was exact (`max_abs_error=0.0`,
  `relative_rmse=0.0`) but slower: current compact-affine
  `0.0005043599987402559s/iter`, execution view
  `0.001509309996617958s/iter`, ratio `0.3341659432922455x`, with
  `0.03125 GiB` extra resident payload bytes.
- Segment transaction was exact (`state_equal_to_serial=true`,
  `tokens_equal_to_serial=true`, `device_status_word=1`) but slower:
  serial `2.523000002838671e-05s/iter`, transaction
  `7.263000006787478e-05s/iter`, ratio `0.3473771169600518x`.
- Device argmax-to-embedding splice passed: token index, embedding table, and
  next embedding were on `cuda:0`, `max_abs_embedding_error=0.0`, no full logits
  were materialized or returned to host, and there was no host token-id
  dependency before the next embedding.

Status:
`PASS_T034_C_DEVICE_SPLICE_IMPLEMENTED_A_B_EXACT_BUT_SPEED_NEGATIVE`.

Claim boundary:

T034 is a real GPU-bound implementation with no CPU offload claim. It is not a
speed win, not a VRAM win, and not a production throughput claim. The full
embedding table needed for the exact device splice raises peak CUDA by
`2.8966827392578125 GiB` in this one-layer smoke. The useful surviving result is
the C-path closure proof; A and B should not be promoted without a different
kernel/state-machine embedding.

---

## R461 T035 Row-Successor Runtime Path One-Layer Smoke (2026-05-27)

Source: `R461` / `T035`

Artifacts:

- `src/transformer_residency/t035_runtime_guardrails.py`
- `scripts/research/t001_transformer_residency_probe.py`
- `.workspace/research/t035-broad-breakthrough-extraordinary/implementation-log.md`
- `.work/jobs/t001-transformer-residency-probe/t035-row-successor-runtime/latest/summary.json`

R461 implements the T035 runtime closeout for `C02-V03` as an opt-in real T017
gate. The CLI emitted `T035_ROW_SUCCESSOR_RUNTIME_PATH_ACTIVE`, completed
`1/1` layer, kept the device-payload path on CUDA, closed the exact device
argmax-to-embedding splice, avoided full-logit materialization and full-logit
host return, and had `0` hot payload load events.

Measured one-layer comparison:

| path | elapsed scope | tokens/s | peak CUDA allocated GiB | peak CUDA reserved GiB | resident CUDA payload GiB |
|---|---|---:|---:|---:|---:|
| R459/T033 one-layer reference | layer-limited smoke | `2.884471992687007` | `0.5345873832702637` | `0.54296875` | `0.5114963054656982` |
| R461/T035 row-successor runtime | layer-limited smoke | `0.7697834145182998` | `3.432966709136963` | `3.44140625` | `0.5114963054656982` |

T035 readiness:

- `t035_runtime_smoke_pass=true`
- `t035_gpu_bound_device_payload_path=true`
- `t035_exact_device_splice_closed=true`
- `t035_no_full_logits_materialized_or_returned=true`
- `t035_hot_payload_free=true`
- `t035_peak_within_splice_cap=false`
- blocker: `peak_above_t035_splice_cap`

Status:
`PASS_T035_C02V03_EXACT_GPU_SPLICE_IMPLEMENTED_FAIL_PEAK_CAP`.

Claim boundary:

R461 is not a speed win, not a VRAM win, and not a production full64 throughput
claim. It is a useful negative closeout: the real before-host-token embedding
splice currently requires a dense full `model.embed_tokens.weight` table on GPU.
That table raises peak CUDA without increasing resident packed payload bytes, so
T035 must not be promoted. The next useful work is a new T036 denominator
breakthrough that avoids this dense sidecar or replaces the whole token loop.

---

## R462 R441 A10-C04 Contractive Green Credit Smoke (2026-06-01)

Source: `R462` / `R441` / `A10-C04`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/train.c`
- `.workspace/training/rounds/r441_a10_c04_contractive_green_credit.ps1`
- `.workspace/research/r441-super-extraordinary-model-frontier/implementation-log.md`
- `.work/jobs/r441-a10-c04-contractive-green-credit/smoke-01/baseline.log`
- `.work/jobs/r441-a10-c04-contractive-green-credit/smoke-01/candidate.log`

R462 implemented `--contractive-green-credit` as an opt-in real v12 training
path. The mechanism applies finite-horizon Green normalization to accepted SSM
state-gradient dimensions only when the local recurrence coefficient satisfies a
contractive tail certificate. The candidate emitted:

```text
[CONTRACTIVE_GREEN_CREDIT] active=1 apply=1 gate=1 horizon=1024 max_alpha=0.980000 tail_budget=0.100000
```

Measured 3-step Vulkan smoke:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44104` | `40.062218` | `510` | `740.14 MB` |
| contractive Green credit | `test_data.bin` | `small_data.bin` | `3` | `1` | `44104` | `40.062160` | `506-508` | `740.14 MB` |

Step-3 gate metrics:

```text
accepted_fraction=1.000000
mean_alpha=0.796622
max_alpha=0.800000
mean_tail_debt=0
max_tail_debt=0
mean_scale=1.000000
min_scale=0.995719
max_scale=1.543158
descent_gain=1.080487
line_shrink=0.910344
net_descent_gain=0.983615
priced_ratio=1.016658
identifying=1
pass=0
```

Status:
`PASS_R462_A10C04_REAL_PATH_ACTIVATION_PRICED_NEGATIVE_SMALL_SMOKE`.

Claim boundary:

R462 is a real opt-in training-path implementation and a valid activation smoke.
It is not a training-speed claim, not a memory claim, and not a quality claim.
The contraction/tail certificate behaved as intended on the small fixture, but
the priced descent proxy did not clear the improvement condition. The tiny
heldout delta is favorable but too small and too low-scale to treat as a model
quality result.

---

## R463 R441 A10-C01 Interior Adjoint Tube Smoke (2026-06-01)

Source: `R463` / `R441` / `A10-C01`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/train.c`
- `.workspace/training/rounds/r441_a10_c01_interior_adjoint_tube.ps1`
- `.workspace/research/r441-super-extraordinary-model-frontier/implementation-log.md`
- `.work/jobs/r441-a10-c01-interior-adjoint-tube/smoke-01/baseline.stdout.log`
- `.work/jobs/r441-a10-c01-interior-adjoint-tube/smoke-01/candidate.stdout.log`
- `.work/jobs/r441-a10-c01-interior-adjoint-tube/apply-diagnostic-01/`

R463 implemented `--interior-adjoint-tube` as an opt-in real v12 training path.
The mechanism selects top-k state-coordinate gradient-energy directions as the
paid interior tube and scales accepted residual coordinates by
`--interior-adjoint-tube-residual-scale`. The candidate emitted:

```text
[INTERIOR_ADJOINT_TUBE] active=1 apply=1 gate=1 rank=32 residual_budget=0.100000 verify_fraction=0.050000 overhead=0.020000
```

Measured 3-step Vulkan smoke:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44101` | `40.062904` | `509-511` | `740.14 MB` |
| interior adjoint tube | `test_data.bin` | `small_data.bin` | `3` | `1` | `44101` | `40.062908` | `509-510` | `740.14 MB` |

Strict step-3 gate metrics:

```text
rank=32
accepted_layer_fraction=0.000000
params=163950
tube_fraction=0.000000
exact_energy=125.50494
tube_energy=98.7175536
residual_energy=26.7873863
residual_fraction=0.154008
energy_covered=0.845992
residual_budget=0.100000
priced_ratio=0.224008
descent_per_paid_work=3.776620
identifying=1
pass=0
```

Apply-path diagnostic:

A permissive one-step diagnostic with residual budget `0.40` and minimum energy
covered `0.60` accepted all six layers and emitted
`accepted_layer_fraction=1.000000`, `tube_fraction=0.249791`,
`residual_fraction=0.276579`, `descent_gain=0.792566`, and `pass=1`. This
proves the opt-in path changes gradients when accepted, but it is not a gain
claim.

Status:
`PASS_R463_A10C01_REAL_PATH_ACTIVATION_STRICT_RESIDUAL_NEGATIVE_SMALL_SMOKE`.

Claim boundary:

R463 is a real opt-in v12 training-path implementation and activation smoke. It
is not a training-speed claim, not a memory claim, and not a quality claim. The
strict interior tube failed because the residual was too diffuse for the
validated A10-C01 budget.

---

## R464 R441 A10-C03 Fisher-SNR Event / Innovation Credit Measure Smoke (2026-06-01)

Source: `R464` / `R441` / `A10-C03`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/gpu/gpu_pipeline.c`
- `src/model_v12/src/train.c`
- `.workspace/training/rounds/r441_a10_c03_event_innovation_credit.ps1`
- `.workspace/research/r441-super-extraordinary-model-frontier/implementation-log.md`
- `.work/jobs/r441-a10-c03-event-innovation-credit/smoke-02/baseline.stdout.log`
- `.work/jobs/r441-a10-c03-event-innovation-credit/smoke-02/candidate.stdout.log`

R464 implemented `--event-innovation-credit` as an opt-in real v12 training
path. It scores sequence positions using forward-owned hidden-state innovation,
gate curvature, and boundary/pre-boundary boost, then prices top event
positions against exact transition-credit energy. The candidate emitted:

```text
[EVENT_INNOVATION_CREDIT] active=1 apply=1 gate=1 event_fraction=0.250000 debt_budget=0.100000 score_cost=0.050000 verify_fraction=0.050000 overhead=0.020000 low_event_scale=0.250000 boundary_boost=2.000000
```

Measured 3-step Vulkan smoke:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44103` | `40.061981` | `508-511` | `740.14 MB` |
| event innovation credit | `test_data.bin` | `small_data.bin` | `3` | `1` | `44103` | `40.061981` | `499-501` | `740.14 MB` |

Strict step-3 gate metrics:

```text
layers=72
applied_layers=0
positions=73728
event_positions=18432
event_fraction=0.250000
exact_energy=0.0806079523
event_energy=0.0182242314
low_event_energy=0.0623837209
low_event_debt=0.773915
energy_covered=0.226085
priced_ratio=1.143915
descent_per_paid_work=0.197641
identifying=1
pass=0
```

Status:
`PASS_R464_A10C03_REAL_PATH_ACTIVATION_EVENT_DEBT_NEGATIVE_SMALL_SMOKE`.

Claim boundary:

R464 is a real opt-in v12 training-path implementation and activation smoke. It
is not a training-speed claim, not a memory claim, and not a quality claim. The
event/innovation selector did not identify CE-relevant transition-credit mass on
the small fixture: only `22.6085%` of exact transition-credit energy landed in
the top `25%` event positions, leaving `77.3915%` low-event debt against a
`10%` budget.

---

## R465 R441 A8-C01 Adaptive Recurrent Credit Horizon Certificate Smoke (2026-06-01)

Source: `R465` / `R441` / `A8-C01`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/gpu/gpu_pipeline.c`
- `src/model_v12/src/train.c`
- `.workspace/training/rounds/r441_a8_c01_adaptive_credit_horizon.ps1`
- `.workspace/research/r441-super-extraordinary-model-frontier/implementation-log.md`
- `.work/jobs/r441-a8-c01-adaptive-credit-horizon/smoke-01/baseline.log`
- `.work/jobs/r441-a8-c01-adaptive-credit-horizon/smoke-01/candidate.log`

R465 implemented `--adaptive-credit-horizon` as an opt-in real v12 training
path. It chooses a recurrent credit horizon, audits omitted transition-credit
rows against exact `dA_per_step` energy, and checks a recurrence-product bound
from `a_eff`. The candidate emitted:

```text
[ADAPTIVE_CREDIT_HORIZON] active=1 apply=1 gate=1 min_horizon=1 max_horizon=128 tail_budget=0.100000 bound_budget=0.100000 verify_fraction=0.050000 overhead=0.020000 min_headroom=0.100000 tail_scale=0.000000
```

Measured 3-step Vulkan smoke:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44181` | `40.063820` | `509-511` | `740.14 MB` |
| adaptive credit horizon | `test_data.bin` | `small_data.bin` | `3` | `1` | `44181` | `40.063816` | `506-509` | `740.14 MB` |

Strict step-3 gate metrics:

```text
layers=72
applied_layers=0
positions=73728
skipped_positions=64512
skipped_fraction=0.875000
avg_horizon=128.000000
horizon_fraction=0.125000
exact_energy=0.080830851
tail_energy=0.0711515268
tail_debt=0.880252
energy_covered=0.119748
mean_bound=3.56953225e-08
max_bound=7.06391745e-08
headroom=-0.955504
priced_ratio=1.075252
descent_per_paid_work=0.111367
identifying=1
pass=0
```

Status:
`PASS_R465_A8C01_REAL_PATH_ACTIVATION_HORIZON_DEBT_NEGATIVE_SMALL_SMOKE`.

Claim boundary:

R465 is a real opt-in v12 training-path implementation and activation smoke. It
is not a training-speed claim, not a memory claim, and not a quality claim. The
small fixture falsifies the useful A8-C01 premise for this setting: even a
short horizon with a tiny recurrence-product bound leaves `88.0252%` exact
transition-credit tail debt, so recurrent work deletion is not safe here.

---

## R466 R441 A8-C02 Certified CE-Debt Skip Ledger Smoke (2026-06-01)

Source: `R466` / `R441` / `A8-C02`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/gpu/gpu_pipeline.c`
- `src/model_v12/src/train.c`
- `.workspace/training/rounds/r441_a8_c02_ce_debt_skip.ps1`
- `.workspace/research/r441-super-extraordinary-model-frontier/implementation-log.md`
- `.work/jobs/r441-a8-c02-ce-debt-skip/smoke-01/baseline.log`
- `.work/jobs/r441-a8-c02-ce-debt-skip/smoke-01/candidate.log`

R466 implemented `--ce-debt-skip` as an opt-in real v12 training path. It
certifies sampled-CE tokens by exact current CE and sampled gradient norm,
prices skipped loss and gradient debt, and only attenuates certified
`ce_d_logits` rows when the certificate passes. The candidate emitted:

```text
[CE_DEBT_SKIP] active=1 apply=1 gate=1 max_loss=40.000000 max_grad_norm=1.000000 max_debt=0.100000
```

Measured 3-step Vulkan smoke:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44182` | `40.061634` | `508-509` | `740.14 MB` |
| CE-debt skip | `test_data.bin` | `small_data.bin` | `3` | `1` | `44182` | `40.061626` | `495-509` | `740.14 MB` |

Strict step-3 gate metrics:

```text
batches=12
applied_batches=0
tokens=12276
certified_tokens=6289
certified_fraction=0.512300
loss_debt=0.378645
grad_debt=0.502117
debt=0.502117
energy_covered=0.497883
priced_ratio=1.059817
descent_per_paid_work=0.469782
identifying=1
pass=0
```

Status:
`PASS_R466_A8C02_REAL_PATH_ACTIVATION_CE_DEBT_NEGATIVE_SMALL_SMOKE`.

Claim boundary:

R466 is a real opt-in v12 training-path implementation and activation smoke. It
is not a training-speed claim, not a memory claim, and not a quality claim. The
strict gate is negative because certified low-loss tokens still carry `50.2117%`
of sampled CE-gradient energy, far above the `10%` debt budget.

---

## R467 R441 A8-C03 Gradient-Coherence / Transition-Quotient Smoke (2026-06-01)

Source: `R467` / `R441` / `A8-C03`

Artifacts:

- `src/model_v12/src/core/types.h`
- `src/model_v12/src/gpu/gpu_pipeline.c`
- `src/model_v12/src/train.c`
- `.workspace/training/rounds/r441_a8_c03_gradient_coherence_quotient.ps1`
- `.workspace/research/r441-super-extraordinary-model-frontier/implementation-log.md`
- `.work/jobs/r441-a8-c03-gradient-coherence-quotient/smoke-01/baseline.log`
- `.work/jobs/r441-a8-c03-gradient-coherence-quotient/smoke-01/candidate.log`

R467 implemented `--gradient-coherence-quotient` as an opt-in real v12 training
path. It groups recurrent `dA_per_step` transition-credit rows by a cheap
forward transition-state signature, computes a weighted representative plus
exact residual ledger per cluster, and only coalesces rows when residual debt
and denominator pricing pass. The candidate emitted:

```text
[GRADIENT_COHERENCE_QUOTIENT] active=1 apply=1 gate=1 bins=64 probe_stride=16 max_residual=0.100000
```

Measured 3-step Vulkan smoke:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44183` | `40.063213` | `509-510` | `740.14 MB` |
| gradient-coherence quotient | `test_data.bin` | `small_data.bin` | `3` | `1` | `44183` | `40.063221` | `502-503` | `740.14 MB` |

Strict step-3 gate metrics:

```text
layers=72
applied_layers=0
positions=73728
nonempty_clusters=4460
avg_cluster_size=16.530941
representative_fraction=0.060493
residual_replay_positions=73689
residual_replay_fraction=0.999471
exact_energy=0.0814618513
representative_energy=0.00597459275
residual_energy=0.0754872585
residual_debt=0.926658
energy_covered=0.073342
saved_fraction=0.939507
priced_ratio=2.106622
descent_per_paid_work=0.034815
identifying=1
pass=0
```

Status:
`PASS_R467_A8C03_REAL_PATH_ACTIVATION_TRANSITION_QUOTIENT_RESIDUAL_NEGATIVE_SMALL_SMOKE`.

Claim boundary:

R467 is a real opt-in v12 training-path implementation and activation smoke. It
is not a training-speed claim, not a memory claim, and not a quality claim. The
strict gate is negative because apparent cluster multiplicity is almost entirely
residual debt: `92.6658%` of exact transition-credit energy remains outside the
representatives and `99.9471%` of positions require residual replay.

---

## R468 R441 A6-C01 Residual-Certified Quotient/Scaffold Smoke (2026-06-01)

Source: `R468` / `R441` / `A6-C01`

R468 implemented `--residual-certified-scaffold` as an opt-in v12 scaffold over
the existing `macro_ce_quotient` substrate. The conservative support-4 smoke
activated the path but produced no quotient object (`contexts=0 classes=0`) and
therefore no gate decision. A support-1 diagnostic produced `contexts=6642`,
`classes=602`, `coverage=0.705969`, `priced_work_ratio=0.404667`, and `pass=1`.

Measured support-1 3-step Vulkan compare:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44161` | `40.064137` | `506-508` | `740.14 MB` |
| residual scaffold | `test_data.bin` | `small_data.bin` | `3` | `1` | `44161` | `40.064045` | `504-506` | `740.14 MB` |

Step-3 support-1 gate:

```text
positions=12264
covered=9238
coverage=0.753262
context_ce=7.451621
quotient_ce=4.710372
ce_excess=-2.741249
audit_budget=0.100000
random_ce=7.426596
random_advantage=2.716224
quotient_ratio=0.090635
residual_price=0.000000
priced_work_ratio=0.357374
net_gain_estimate=2.798191
training_rows=12276
training_covered=8872
training_coverage=0.722711
training_effective_weight=0.999023
original_ce_visible=1
identifying=1
pass=1
```

Status:
`PASS_R468_A6C01_REAL_PATH_ACTIVATION_SCAFFOLD_QUOTIENT_POSITIVE_SMALL_GATE`.

Claim boundary:

R468 is a real opt-in training-path implementation and positive small gate for
scaffold coverage on an identifying support-1 fixture. It is not a headline
quality, speed, or memory claim; the support-4 fixture remains non-identifying
and the heldout movement is too small and short-run to promote as a quality win.

---

## R469 R441 A6-C03 Source-Certified Minimal Capability Witness Gate (2026-06-01)

Source: `R469` / `R441` / `A6-C03`

R469 implemented
`scripts/research/r441_a6_c03_source_certified_witness_gate.py` and the preset
`.workspace/training/rounds/r441_a6_c03_source_certified_witness.ps1`. The gate
extracts deterministic source-certified witness triples with source hashes,
source pointers, and local verifiers, then compares raw examples against
witness learning under equal paid budget on source and renamed variants.

Diagnostic smoke:

```text
[SOURCE_CERTIFIED_WITNESS_GATE] active=1 decision=PASS_R469_A6C03_SOURCE_CERTIFIED_WITNESS_GATE false_rate=0.000000 raw_source_acc=0.799107 witness_source_acc=1.000000 raw_renamed_acc=0.000000 witness_renamed_acc=1.000000 priced_gain=2.502793 rename_transfer_ratio=1.000000
```

Measured metrics:

```text
sources=48
witnesses_available=144
witnesses_verified=144
witnesses_used=144
paid_budget=220.0
raw_spent=220.0
witness_spent=116.4
raw_source_accuracy=0.7991071428571429
witness_source_accuracy=1.0
raw_renamed_accuracy=0.0
witness_renamed_accuracy=1.0
source_gain=1.2513966480446927
priced_gain=2.5027932960893855
source_ce_delta=0.2242602460142848
rename_transfer_ratio=1.0
false_rate=0.0
```

False-witness control:

```text
decision=FAIL_R469_A6C03_SOURCE_CERTIFIED_WITNESS_GATE
injected_false=8
false_rate=0.05555555555555558
rho_false_within_budget=false
```

Status:
`PASS_R469_A6C03_SOURCE_CERTIFIED_WITNESS_GATE`.

Claim boundary:

R469 is a positive CPU source-certified capability-transfer gate with a failing
false-witness control. It is not a v12 production training-speed claim, memory
claim, or broad quality claim.

---

## R470 R441 A6-C02 Fisher-SNR / Equal-Area Gradient Geometry Sampling Smoke (2026-06-01)

Source: `R470` / `R441` / `A6-C02`

R470 implemented `--equal-area-fisher-sampling` as an opt-in v12 gate over the
variance-priced transition-credit sampler. Build passed with known warnings.
The implementation was corrected so the sampled transition-credit update cannot
apply when the stricter A6 equal-area certificate fails.

Measured 3-step Vulkan smoke:

| path | train dataset | heldout dataset | steps | batch | seed | final heldout loss | train log tok/s | train-start Vulkan allocation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `test_data.bin` | `small_data.bin` | `3` | `1` | `44162` | `40.063713` | `508-509` | `740.14 MB` |
| equal-area Fisher | `test_data.bin` | `small_data.bin` | `3` | `1` | `44162` | `40.063717` | `507-509` | `740.14 MB` |

Strict step-3 gate:

```text
sampled_fraction=0.247287
ess_fraction=0.859290
ess_floor=0.250000
variance_proxy=0.596385
bias_budget=0.100000
alignment=0.989334
priced_work=0.367287
descent_per_paid_work=1.069869
min_descent_per_paid_work=1.100000
training_layers=72
applied_layers=0
fallback_layers=72
identifying=1
pass=0
```

Status:
`PASS_R470_A6C02_REAL_PATH_ACTIVATION_EQUAL_AREA_FISHER_NEGATIVE_SMALL_SMOKE`.

Claim boundary:

R470 is a real opt-in v12 training-path gate, but it is negative on this small
fixture. No quality, speed, or memory claim is allowed because the variance/bias
proxy exceeds budget and no sampled layers were permitted to apply.

---

## R471 R441 A7-C03 Reversible Structure / Local Gauge Quotient Gate (2026-06-01)

Source: `R471` / `R441` / `A7-C03`

R471 implemented `scripts/research/r441_a7_c03_reversible_gauge_quotient_gate.py`
and `.workspace/training/rounds/r441_a7_c03_reversible_gauge_quotient.ps1`. The
gate builds a deterministic canonical local-gauge stream plus sidecar table and
requires exact inverse rendering before the proxy loss bound can pass.

Diagnostic:

```text
decision=PASS_R471_A7C03_REVERSIBLE_GAUGE_QUOTIENT_GATE
spans=64
transformed_spans=48
exact_roundtrip=1
roundtrip_failures=0
coverage=0.830948121645796
sidecar_density=0.09755113024757804
loss_bound_ratio=0.8844364937388193
priced_gain=1.130663430420712
bpe_expansion=0.8844364937388193
total_original_proxy_loss=8944.0
total_paid_proxy_loss=7910.4
```

False-control:

```text
decision=FAIL_R471_A7C03_REVERSIBLE_GAUGE_QUOTIENT_GATE
exact_roundtrip=0
roundtrip_failures=1
```

Status:
`PASS_R471_A7C03_REVERSIBLE_GAUGE_QUOTIENT_GATE`.

Claim boundary:

R471 is a positive CPU reversible dataset-object gate with a failing inverse
audit control. It proves exact map-back and a priced proxy bound on the
generated structured/gauge corpus only. It is not a production v12 speed,
memory, or broad quality claim.

---

## R472 R441 A7-C01 Reversible Causal Episode / Document Transaction Tape Gate (2026-06-01)

Source: `R472` / `R441` / `A7-C01`

R472 implemented `scripts/research/r441_a7_c01_reversible_episode_tape_gate.py`
and `.workspace/training/rounds/r441_a7_c01_reversible_episode_tape.ps1`. The
gate parses deterministic transaction documents into causal episodes plus
payload residuals, verifies exact rendering back to source text, and compares a
flat heldout CE proxy against the episode-event plus payload-residual bound.

Diagnostic:

```text
decision=PASS_R472_A7C01_REVERSIBLE_EPISODE_TAPE_GATE
docs=72
parsed_docs=60
fallback_docs=12
exact_roundtrip=1
sync_errors=0
coverage=0.8646616541353384
residual_density=0.1608695652173913
episode_token_ratio=0.7086956521739131
flat_ce_proxy=4.208189710708242
episode_ce_bound_proxy=3.208846403898699
ce_gain=1.3114338241915715
```

False-control:

```text
decision=FAIL_R472_A7C01_REVERSIBLE_EPISODE_TAPE_GATE
exact_roundtrip=0
sync_errors=1
```

Status:
`PASS_R472_A7C01_REVERSIBLE_EPISODE_TAPE_GATE`.

Claim boundary:

R472 is a positive CPU reversible episode-object gate with a failing
sync-error control and a heldout proxy CE advantage on the generated
transaction corpus. It is not a production v12 speed, memory, or broad quality
claim.

---

## R473 R441 A7-C02 Predictive Sufficient-State Quotient / Weighted Transition Measure Gate (2026-06-01)

Source: `R473` / `R441` / `A7-C02`

R473 implemented
`scripts/research/r441_a7_c02_predictive_transition_quotient_gate.py` and
`.workspace/training/rounds/r441_a7_c02_predictive_transition_quotient.ps1`.
The gate builds deterministic transition cells, charges residual exceptions,
and compares heldout original-token CE proxy against a local-prefix baseline
while enforcing low KL debt.

Diagnostic:

```text
decision=PASS_R473_A7C02_PREDICTIVE_TRANSITION_QUOTIENT_GATE
docs=96
cells=22
coverage=0.8064337215751525
residual_mass=0.19356627842484747
kl_proxy=0.0
hard_miss_rate=0.0
prev_prefix_ce_proxy=3.236340786622726
quotient_ce_bound_proxy=2.123780988113697
ce_gain=1.5238580648078894
covered_tokens=1454
residual_tokens=349
```

False-control:

```text
decision=FAIL_R473_A7C02_PREDICTIVE_TRANSITION_QUOTIENT_GATE
cells=7
kl_proxy=0.5809345073305134
```

Status:
`PASS_R473_A7C02_PREDICTIVE_TRANSITION_QUOTIENT_GATE`.

Claim boundary:

R473 is a positive CPU predictive transition-object gate with a collapsed-cell
KL false-control. It is not a production v12 speed, memory, or broad quality
claim.

---

## R474 R441 A1-C02 Prospective Query Register Gate (2026-06-01)

Source: `R474` / `R441` / `A1-C02`

R474 implemented `scripts/research/r441_a1_c02_prospective_query_register_gate.py`
and `.workspace/training/rounds/r441_a1_c02_prospective_query_register.ps1`.
The gate writes exact spans only for declared instruction-before-evidence
queries and rejects ambiguous acceptors, overflow, checksum failure, misses,
false accepts, or dense-cache-like memory.

Diagnostic:

```text
decision=PASS_R474_A1C02_PROSPECTIVE_QUERY_REGISTER_GATE
records=192
queries=8
gap_lines=96
exact_accuracy=1.0
ambiguous_acceptors=0
overflows=0
checksum_failures=0
memory_ratio=0.06944444444444445
stored_token_ratio=0.041666666666666664
dense_cache_bits=36864.0
total_register_bits=2560.0
```

False-control:

```text
decision=FAIL_R474_A1C02_PROSPECTIVE_QUERY_REGISTER_GATE
exact_accuracy=0.875
ambiguous_acceptors=1
overflows=1
```

Status:
`PASS_R474_A1C02_PROSPECTIVE_QUERY_REGISTER_GATE`.

Claim boundary:

R474 is a positive CPU exact-copy gate for the prospective declared-query
domain only. It is not general arbitrary long-context recall, production v12
integration, speed, memory, or broad quality claim.

---

## R475 R441 A1-C04 Dictionary Syndrome Quotient Gate (2026-06-01)

Source: `R475` / `R441` / `A1-C04`

R475 implemented `scripts/research/r441_a1_c04_dictionary_syndrome_quotient_gate.py`
and `.workspace/training/rounds/r441_a1_c04_dictionary_syndrome_quotient.ps1`.
The gate compiles a finite model-owned dictionary, encodes structured spans as
descriptors plus sparse literal exceptions, and requires exact reconstruction
and checksum verification while charging dictionary, descriptor, exception, and
verification bits.

Diagnostic:

```text
decision=PASS_R475_A1C04_DICTIONARY_SYNDROME_QUOTIENT_GATE
spans=96
accepted_spans=96
exact_decode_failures=0
checksum_failures=0
raw_bits=24192.0
paid_bits=4753.0
dictionary_bits=1712.0
descriptor_bits=1056
exception_bits=449.0
verify_bits=1536.0
exception_density=0.01455026455026455
compression_gain=5.089837997054492
dictionary_entry_ratio=0.041666666666666664
```

False control:

```text
decision=FAIL_R475_A1C04_DICTIONARY_SYNDROME_QUOTIENT_GATE
exact_decode_failures=1
checksum_failures=1
accepted_spans=95
```

Status:
`PASS_R475_A1C04_DICTIONARY_SYNDROME_QUOTIENT_GATE`.

Claim boundary:

R475 is a positive CPU exact-copy gate for finite structured spans with a
model-owned dictionary and sparse exceptions. It is not arbitrary corpus lookup,
general long-context recall, production v12 integration, training speed,
runtime memory, or broad quality evidence.

---

## R476 R441 A1-C03 Sparse Exactness Plane Gate (2026-06-01)

Source: `R476` / `R441` / `A1-C03`

R476 implemented `scripts/research/r441_a1_c03_sparse_exactness_plane_gate.py`
and `.workspace/training/rounds/r441_a1_c03_sparse_exactness_plane.ps1`.
The gate instantiates A1-C03 as a finite-field keyed slot ledger with rank and
collision certificates, priced as an oracle frontier over sparse late
exact-copy obligations.

Diagnostic:

```text
decision=PASS_R476_A1C03_SPARSE_EXACTNESS_PLANE_GATE
records=256
queries=40
active_entries=12
accepted_exact=12
accepted_exact_mass=0.3
false_accept_rate=0.0
collision_failures=0
rank=12
rank_pass=true
dense_context_bits=32768.0
memory_bits=2808.0
memory_ratio=0.085693359375
baseline_failure_bits=20480.0
candidate_total_bits=17784.0
total_cost_ratio=0.868359375
```

False control:

```text
decision=FAIL_R476_A1C03_SPARSE_EXACTNESS_PLANE_GATE
collision_failures=1
active_entries=11
accepted_exact_mass=0.275
```

Status:
`PASS_R476_A1C03_SPARSE_EXACTNESS_PLANE_GATE`.

Claim boundary:

R476 is a positive CPU oracle-frontier gate for a concrete sparse exactness
plane. It is not a trained promotion policy, arbitrary late recall, production
v12 integration, training speed, runtime memory, or broad quality evidence.

---

## R477 R441 A2-C02 Constraint Debt Relation Ledger Gate (2026-06-01)

Source: `R477` / `R441` / `A2-C02`

R477 implemented `scripts/research/r441_a2_c02_constraint_debt_relation_ledger_gate.py`
and `.workspace/training/rounds/r441_a2_c02_constraint_debt_relation_ledger.ps1`.
The gate uses an observer-only typed obligation ledger over synthetic
scope/delimiter, definition-use, quote/coreference, and copy-with-distractor
relations, with compact debt notes and exact pair-oracle settlement.

Diagnostic:

```text
decision=PASS_R477_A2C02_CONSTRAINT_DEBT_RELATION_LEDGER_GATE
relations=96
events=384
true_demands=96
settled_true=96
constraint_coverage=1.0
false_settlements=0
false_debt_fraction=0.0
ledger_payload_entropy_ratio=0.475
baseline_relation_ce=18432.0
paid_cost=8832.0
paid_ratio=0.4791666666666667
saved_ce=9216.0
```

False control:

```text
decision=FAIL_R477_A2C02_CONSTRAINT_DEBT_RELATION_LEDGER_GATE
false_settlements=16
false_debt_fraction=0.2222222222222222
paid_ratio=0.59375
```

Status:
`PASS_R477_A2C02_CONSTRAINT_DEBT_RELATION_LEDGER_GATE`.

Claim boundary:

R477 is a positive CPU observer-only relation-obligation gate on a synthetic
exact-pair suite. It is not production v12 integration, not a replacement for
attention on arbitrary pair access, and not a training speed, runtime memory, or
broad quality claim.

---

## R478 R441 A2-C01 Pair Residual Moment Gate (2026-06-01)

Source: `R478` / `R441` / `A2-C01`

R478 implemented `scripts/research/r441_a2_c01_pair_residual_moment_gate.py`
and `.workspace/training/rounds/r441_a2_c01_pair_residual_moment.ps1`.
The gate uses a frozen pair-residual moment observer on a synthetic
relation-heavy corpus, carries aggregate relation moments by type and compact
bucket, compares against shuffled-moment and same-width generic controls, and
charges moment state, update, verify, residual repair, and false-interaction
harm.

Diagnostic:

```text
decision=PASS_R478_A2C01_PAIR_RESIDUAL_MOMENT_GATE
relations=128
events=384
captured_useful_pair_mass=128.0
useful_pair_mass_exact=128.0
captured_fraction=1.0
residual_fraction=0.0
false_interaction_harm=4.040000000000003
false_harm_fraction=0.03156250000000002
shuffled_captured_fraction=0.0
generic_captured_fraction=0.0
control_advantage=1.0
moment_width=32
state_width_ratio=0.08333333333333333
baseline_cost=8192.0
paid_cost=1732.04
paid_ratio=0.2114306640625
```

False control:

```text
decision=FAIL_R478_A2C01_PAIR_RESIDUAL_MOMENT_GATE
candidate_observer=shuffled
captured_fraction=0.0
residual_fraction=1.0
false_harm_fraction=8.160000000000002
paid_ratio=0.96193359375
```

Status:
`PASS_R478_A2C01_PAIR_RESIDUAL_MOMENT_GATE`.

Claim boundary:

R478 is a positive CPU frozen-observer gate on a low-rank synthetic relation
fixture. It is not a trained moment channel, production v12 integration,
arbitrary pair access, training speed, runtime memory, or broad quality
evidence.

---

## R479 R441 A2-C04 Relation Quotient Graph Gate (2026-06-01)

Source: `R479` / `R441` / `A2-C04`

R479 implemented `scripts/research/r441_a2_c04_relation_quotient_graph_gate.py`
and `.workspace/training/rounds/r441_a2_c04_relation_quotient_graph.ps1`.
The gate uses a relation-native quotient graph over synthetic segment pairs:
typed cells carry multiplicity and a representative effect, with residual
replay, verification, split audit, update, edge, memory, and false-equivalence
costs charged against an exact pair baseline.

Diagnostic:

```text
decision=PASS_R479_A2C04_RELATION_QUOTIENT_GRAPH_GATE
pairs=216
total_useful_pair_mass=226.8
quotient_cells=36
quotient_edges=36
accepted_mass_fraction=1.0
high_mass_cell_residual=0.009570227278326737
residual_fraction=0.005818887056566409
split_gain_fraction=0.0
state_width_ratio=0.16666666666666666
baseline_cost=14515.2
paid_cost=1380.4623094034728
paid_ratio=0.09510460134228069
```

False control:

```text
decision=FAIL_R479_A2C04_RELATION_QUOTIENT_GRAPH_GATE
candidate_signature=broad
accepted_mass_fraction=0.11640211640211641
high_mass_cell_residual=0.2607955179179158
residual_fraction=0.22090028511495208
split_gain_fraction=0.2150813980583857
paid_ratio=0.445605715535864
```

Status:
`PASS_R479_A2C04_RELATION_QUOTIENT_GRAPH_GATE`.

Claim boundary:

R479 is a positive CPU relation-native segment-pair quotient graph gate on a
synthetic repeated-relation fixture. It is not production v12 integration,
arbitrary pair attention replacement, a trained quotient policy, training
speed, runtime memory, or broad quality evidence.

---

## R480 R441 A3-C03 Coherence-Priced Binding Matrix Gate (2026-06-01)

Source: `R480` / `R441` / `A3-C03`

R480 implemented `scripts/research/r441_a3_c03_coherence_priced_binding_matrix_gate.py`
and `.workspace/training/rounds/r441_a3_c03_coherence_priced_binding_matrix.ps1`.
The gate uses a fixed-state matrix binder `M <- M + k v^T` on fresh renamed
key/value tasks with known key coherence, a surface-repetition control, reset
audit, ordinary-text harm audit, and priced state/update/read/audit/interference
costs.

Diagnostic:

```text
decision=PASS_R480_A3C03_COHERENCE_PRICED_BINDING_MATRIX_GATE
active_bindings=24
key_dim=64
value_dim=32
rank_used=24
rank_budget=32
rank_prompt_slope=0.0
max_key_coherence=0.0
candidate_ce=0.01034564029717977
baseline_ce=3.4657359027997265
ce_improvement=0.9970148792096876
accuracy=1.0
surface_repetition_accuracy=0.0
observed_interference=0.0
predicted_interference=0.0
reset_residual=0.0
ordinary_text_harm=0.0
baseline_cost=6144.0
paid_cost=4416.0
paid_ratio=0.71875
```

False control:

```text
decision=FAIL_R480_A3C03_COHERENCE_PRICED_BINDING_MATRIX_GATE
candidate_profile=high
max_key_coherence=0.9928089887640449
candidate_ce=3.123108138195436
ce_improvement=0.09886147537309616
observed_interference=22.67040282792577
predicted_interference=22.670402827925766
paid_ratio=0.8368250147287801
```

Status:
`PASS_R480_A3C03_COHERENCE_PRICED_BINDING_MATRIX_GATE`.

Claim boundary:

R480 is a positive CPU fixed-state coherence-priced binding matrix gate on a
controlled renamed-symbol fixture. It is not production v12 integration,
general in-context learning, a prompt/KV cache, persistent weight adaptation,
training speed, runtime memory, or broad quality evidence.

---

## R481 R441 A3-C01 Resettable Binding Ledger Gate (2026-06-01)

Source: `R481` / `R441` / `A3-C01`

R481 implemented `scripts/research/r441_a3_c01_resettable_binding_ledger_gate.py`
and `.workspace/training/rounds/r441_a3_c01_resettable_binding_ledger.ps1`. The
gate uses a fixed-width reversible temporary ledger `B`, writes fresh heldout
key/value bindings, reads delayed renamed queries, supports exact erase and hard
reset, audits ordinary-text harm and prompt-cache collapse, and charges state,
write/read/erase/reset/audit/interference costs.

Diagnostic:

```text
decision=PASS_R481_A3C01_RESETTABLE_BINDING_LEDGER_GATE
active_bindings=20
prompt_tokens=109
key_dim=64
value_dim=32
state_width=2048
token_table_entries=0
state_prompt_slope=0.0
candidate_ce=0.01034564029717977
baseline_ce=3.4657359027997265
ce_improvement=0.9970148792096876
accuracy=1.0
surface_repetition_accuracy=0.0
observed_interference=0.0
predicted_interference=0.0
erase_residual=0.0
remaining_accuracy_after_erase=1.0
reset_residual=0.0
post_reset_leak_rate=0.0
ordinary_text_harm=0.0
baseline_cost=5120.0
paid_cost=4388.0
paid_ratio=0.85703125
```

False control:

```text
decision=FAIL_R481_A3C01_RESETTABLE_BINDING_LEDGER_GATE
mode=false-control
accuracy=0.05
candidate_ce=7.61034564029718
ce_improvement=-1.195881582941537
observed_interference=1.9
predicted_interference=0.0
erase_residual=1.131370849898476
remaining_accuracy_after_erase=0.0
paid_ratio=0.86890625
```

Status:
`PASS_R481_A3C01_RESETTABLE_BINDING_LEDGER_GATE`.

Claim boundary:

R481 is a positive CPU resettable binding ledger gate on a controlled
fresh-symbol dictionary/glossary fixture. It is not production v12 integration,
general in-context learning, a prompt/KV cache, persistent weight adaptation,
training speed, runtime memory, or broad quality evidence.

---

## R482 R441 A3-C02 Task-State Transaction Algebra Gate (2026-06-01)

Source: `R482` / `R441` / `A3-C02`

R482 implemented
`scripts/research/r441_a3_c02_task_state_transaction_algebra_gate.py` and
`.workspace/training/rounds/r441_a3_c02_task_state_transaction_algebra.ps1`.
The gate compiles local affine unit-conversion/DSL transactions and scoped
variable writes into a compact task state, then tests unseen compositions,
rollback, empty-stack identity, disjoint-transaction commutation, scope detector
cost, and lookup-only controls.

Diagnostic:

```text
decision=PASS_R482_A3C02_TASK_STATE_TRANSACTION_ALGEBRA_GATE
composition_cases=32
scope_cases=16
query_count=64
transaction_count=96
candidate_ce=0.01001357522654267
lookup_control_ce=3.668500801898679
ce_improvement=0.9972703903400103
composition_accuracy=1.0
lookup_composition_accuracy=0.03125
scope_accuracy=1.0
lookup_scope_accuracy=0.5
rollback_leakage_rate=0.0
empty_stack_identity_error=0.0
disjoint_commutator_residual=0.0
state_width=128
paid_cost=1392.0
baseline_cost=4096.0
paid_ratio=0.33984375
```

False control:

```text
decision=FAIL_R482_A3C02_TASK_STATE_TRANSACTION_ALGEBRA_GATE
candidate_mode=lookup-only
candidate_ce=3.668500801898679
lookup_control_ce=3.668500801898679
ce_improvement=0.0
composition_accuracy=0.03125
lookup_composition_accuracy=0.03125
scope_accuracy=0.5
rollback_leakage_rate=0.5
empty_stack_identity_error=1.0
paid_ratio=0.32285156249999997
```

Status:
`PASS_R482_A3C02_TASK_STATE_TRANSACTION_ALGEBRA_GATE`.

Claim boundary:

R482 is a positive CPU task-state transaction algebra gate on generated
local-rule, unit-conversion, DSL-composition, and variable-shadowing fixtures.
It is not production v12 integration, arbitrary in-context learning, a prompt/KV
cache, persistent adapter/fine-tuning, training speed, runtime memory, or broad
quality evidence.

---

## R483 R441 A4-C03 Modular Residue Arithmetic Lane Gate (2026-06-01)

Source: `R483` / `R441` / `A4-C03`

R483 implemented `scripts/research/r441_a4_c03_modular_residue_arithmetic_gate.py`
and `.workspace/training/rounds/r441_a4_c03_modular_residue_arithmetic.ps1`.
The gate evaluates generated heldout arithmetic expressions through coprime
modular residue lanes and reconstructs exact integer results by CRT only when
`M = product(p_i) > 2N` certifies uniqueness over the bounded result range.

Diagnostic:

```text
decision=PASS_R483_A4C03_MODULAR_RESIDUE_ARITHMETIC_GATE
expressions=128
lane_count=4
crt_product=215441
range_bound=10112
crt_margin=10.652739319620252
crt_range_ok=true
range_violations=0
exact_accuracy=1.0
residue_consistency=1.0
candidate_digit_ce=0.012293977324588428
baseline_digit_ce=9.390229832366328
digit_ce_improvement=0.998690769284239
coverage=1.0
nonarith_ce_harm=0.0
paid_cost=1728.0
baseline_cost=4096.0
paid_ratio=0.421875
```

False control:

```text
decision=FAIL_R483_A4C03_MODULAR_RESIDUE_ARITHMETIC_GATE
candidate_mode=insufficient-range-control
lane_count=2
crt_product=323
range_bound=10112
crt_margin=0.015971123417721517
crt_range_ok=false
range_violations=126
exact_accuracy=0.015625
residue_consistency=1.0
digit_ce_improvement=-2.4490416694266304
paid_ratio=0.2578125
```

Status:
`PASS_R483_A4C03_MODULAR_RESIDUE_ARITHMETIC_GATE`.

Claim boundary:

R483 is a positive CPU bounded-arithmetic residue-lane gate on generated
arithmetic expressions. It is not production v12 integration, a general
reasoning/math solver, an external calculator, a prompt/KV cache, training
speed, runtime memory, or broad quality evidence.

---

## R484 R441 A4-C02 Stack Scope Debt Ledger Gate (2026-06-01)

Source: `R484` / `R441` / `A4-C02`

R484 implemented `scripts/research/r441_a4_c02_stack_scope_debt_ledger_gate.py`
and `.workspace/training/rounds/r441_a4_c02_stack_scope_debt_ledger.ps1`. The
gate tests a finite type-state automaton over bounded code/formal-structure
snippets with a compact ledger for scope closure, definition-use, value kind,
and control phase. Parser-filter and bracket-only controls are priced and
reported separately.

Diagnostic:

```text
decision=PASS_R484_A4C02_STACK_SCOPE_DEBT_LEDGER_GATE
cases=96
query_count=288
corruption_count=288
syntax_valid_continuation_accuracy=1.0
scope_closure_accuracy=1.0
definition_use_accuracy=1.0
control_phase_accuracy=1.0
parser_filter_accuracy=0.4444444444444444
bracket_only_accuracy=0.0
nonbracket_accuracy=1.0
parser_nonbracket_accuracy=0.16666666666666666
nonbracket_improvement=0.8333333333333334
coverage_error_mass=1.0
false_debt_valid=0.0
candidate_ce=0.0010058818149270622
parser_filter_ce=4.445450326259365
ce_improvement_vs_parser=0.9997737278023364
error_reduction_vs_parser=1.0
general_ce_harm=0.0
paid_cost=806.4
baseline_cost=2304.0
paid_ratio=0.35
```

False control:

```text
decision=FAIL_R484_A4C02_STACK_SCOPE_DEBT_LEDGER_GATE
candidate_mode=bracket-only-control
syntax_valid_continuation_accuracy=0.0
scope_closure_accuracy=0.0
definition_use_accuracy=0.0
control_phase_accuracy=0.0
coverage_error_mass=0.0
ce_improvement_vs_parser=-0.7998189822418714
error_reduction_vs_parser=-0.7999999999999999
paid_ratio=0.35
```

Status:
`PASS_R484_A4C02_STACK_SCOPE_DEBT_LEDGER_GATE`.

Claim boundary:

R484 is a positive CPU bounded code-structure/type-state gate on generated
Python-like scope, definition-use, and control-phase snippets. It is not
production v12 integration, general code execution, a parser repair loop,
broad reasoning, training speed, runtime memory, or broad quality evidence.

---

## R485 R441 A4-C04 Execution Trace Homomorphism Gate (2026-06-01)

Source: `R485` / `R441` / `A4-C04`

R485 implemented
`scripts/research/r441_a4_c04_execution_trace_homomorphism_gate.py` and
`.workspace/training/rounds/r441_a4_c04_execution_trace_homomorphism.ps1`. The
gate tests a restricted executable trace homomorphism over a tiny deterministic
language with integer store, branch, capped loop, and print effects. It charges
interpreter, projection, trace-score, transaction, replay, false-accept, and
memory costs.

Diagnostic:

```text
decision=PASS_R485_A4C04_EXECUTION_TRACE_HOMOMORPHISM_GATE
programs=80
trace_steps=420
accepted_mass=1.0
mean_trace_residual=0.0
trace_residual_ce=0.0
low_residual_count=80
low_residual_output_wrong_rate=0.0
output_accuracy=1.0
baseline_output_accuracy=0.25
wrong_rate_reduction=1.0
candidate_output_ce=0.020913916994650165
baseline_output_ce=6.020913916994642
output_ce_improvement=0.9965264547404309
rollback_accuracy=1.0
false_commit_rate=0.0
commit_rate=1.0
abstain_rate=0.0
rollback_ce_gain=8.000000000000002
projection_stability=1.0
paid_cost=730.0
baseline_cost=1920.0
paid_ratio=0.3802083333333333
```

False control:

```text
decision=FAIL_R485_A4C04_EXECUTION_TRACE_HOMOMORPHISM_GATE
candidate_mode=always-commit-control
accepted_mass=0.0
mean_trace_residual=72.25
trace_residual_ce=1.0
output_accuracy=0.0
wrong_rate_reduction=-0.3333333333333333
rollback_accuracy=0.0
false_commit_rate=1.0
output_ce_improvement=-0.33217548491347887
paid_ratio=0.7135416666666666
```

Status:
`PASS_R485_A4C04_EXECUTION_TRACE_HOMOMORPHISM_GATE`.

Claim boundary:

R485 is a positive CPU restricted executable-trace homomorphism and
transactional rollback gate on a tiny deterministic language. It is not
production v12 integration, general code execution, natural-code transfer,
an inference-time external executor or repair loop, training speed, runtime
memory, or broad quality evidence.

---

## R486 R441 A5-C03 Orthogonal Control Plane Gate (2026-06-01)

Source: `R486` / `R441` / `A5-C03`

R486 implemented
`scripts/research/r441_a5_c03_orthogonal_control_plane_gate.py` and
`.workspace/training/rounds/r441_a5_c03_orthogonal_control_plane.ps1`. The gate
tests an observer-first orthogonal control/payload state split for instruction
noninterference, with same-width mixed-plane false-control pricing.

Diagnostic:

```text
decision=PASS_R486_A5C03_ORTHOGONAL_CONTROL_PLANE_GATE
cases=128
neutral_control_ratio=0.04
instruction_payload_ratio=0.03
cross_jacobian_energy=0.002
orthogonality_score=0.8
payload_ce_harm=0.001998002662673056
fake_instruction_leakage=0.01
fake_instruction_rejection=0.99
control_surgery_success=1.0
payload_surgery_harm=0.0
instruction_surgery_payload_harm=0.0
state_surgery_selectivity=1.0
selectivity_ratio=43.640325656881565
generic_same_width_match=false
generic_advantage=0.6799999999999999
saved_drift_ce=87.03999999999999
content_payload_accuracy=0.998
instruction_retention_accuracy=0.95
paid_cost=592.384
baseline_cost=4096.0
paid_ratio=0.144625
```

False control:

```text
decision=FAIL_R486_A5C03_ORTHOGONAL_CONTROL_PLANE_GATE
candidate_mode=mixed-plane-control
neutral_control_ratio=0.72
instruction_payload_ratio=0.46
cross_jacobian_energy=0.38
payload_ce_harm=0.27002713721306015
fake_instruction_leakage=0.42
state_surgery_selectivity=0.11699999999999999
selectivity_ratio=0.2413567250038968
generic_advantage=0.0
saved_drift_ce=0.0
paid_ratio=0.760625
```

Status:
`PASS_R486_A5C03_ORTHOGONAL_CONTROL_PLANE_GATE`.

Claim boundary:

R486 is a positive CPU observer-first noninterference/control-plane gate on
generated instruction-control fixtures. It is not production v12 integration,
a policy/safety judge, refusal-style behavior, broad instruction following,
training speed, runtime memory, or broad quality evidence.

---

## R487 R441 A5-C02 Authority-Scope Lattice Gate (2026-06-01)

Source: `R487` / `R441` / `A5-C02`

R487 implemented
`scripts/research/r441_a5_c02_authority_scope_lattice_gate.py` and
`.workspace/training/rounds/r441_a5_c02_authority_scope_lattice.ps1`. The gate
tests a finite projected authority/scope lattice with scope close rollback,
lower-priority override prevention, feasibility debt, and valid repair.

Diagnostic:

```text
decision=PASS_R487_A5C02_AUTHORITY_SCOPE_LATTICE_GATE
cases=96
transitions=672
transition_accuracy=1.0
lower_priority_override_rate=0.0
scope_leak_after_close=0.0
mean_repair_reduction=0.8499999999999995
false_conflict_debt_rate=0.0
valid_subtask_preservation=1.0
prediction_lead_accuracy=1.0
candidate_ce=0.024485295264218346
recency_ce=6.02448529526419
delimiter_baseline_ce=2.5959138666927695
ce_improvement_vs_recency=0.995935703373122
delimiter_advantage=0.990567755125322
violation_rate=0.0
repair_score=0.8499999999999995
saved_violation_ce=2534.4
paid_cost=2304.0
baseline_cost=12288.0
paid_ratio=0.1875
```

False control:

```text
decision=FAIL_R487_A5C02_AUTHORITY_SCOPE_LATTICE_GATE
candidate_mode=recency-delimiter-control
transition_accuracy=0.0
lower_priority_override_rate=1.0
scope_leak_after_close=1.0
mean_repair_reduction=0.19999999999999962
false_conflict_debt_rate=0.649999999999999
valid_subtask_preservation=0.0
prediction_lead_accuracy=0.0
delimiter_advantage=-1.32075700683377
saved_violation_ce=0.0
paid_ratio=1.9375
```

Status:
`PASS_R487_A5C02_AUTHORITY_SCOPE_LATTICE_GATE`.

Claim boundary:

R487 is a positive CPU finite authority/scope lattice gate on deterministic
nested instruction-control fixtures. It is not production v12 integration, a
policy/refusal classifier, external judging, general instruction following,
training speed, runtime memory, or broad quality evidence.

---

## R488 R441 A5-C04 Ternary Control Syndrome Gate (2026-06-01)

Source: `R488` / `R441` / `A5-C04`

R488 implemented
`scripts/research/r441_a5_c04_ternary_control_syndrome_gate.py` and
`.workspace/training/rounds/r441_a5_c04_ternary_control_syndrome.ps1`. The gate
tests a finite ternary control code for discrete instruction modes, with
minimum-distance certificate, syndrome lead, correction, and authorized-update
checks.

Diagnostic:

```text
decision=PASS_R488_A5C04_TERNARY_CONTROL_SYNDROME_GATE
cases=144
mode_count=6
width=18
min_distance=9
correction_radius=4
tail_threshold=5
binomial_tail_bound=0.011587838005082101
decode_error_rate=0.0
flip_tail_rate=0.0
correlated_tail_rate=0.0
correction_trials=57
correction_success_rate=1.0
authorized_update_trials=36
missed_authorized_update_rate=0.0
syndrome_true_rate=1.0
syndrome_false_rate=0.0
syndrome_lead=1.0
repair_loop_rate=0.0
candidate_ce=0.012317586970644235
uncoded_control_ce=2.0123175869706476
ce_improvement_vs_uncoded=0.9938789050742297
capacity_harm=0.0020000000000000013
saved_control_ce=1566.72
paid_cost=1443.4560000000001
baseline_cost=18432.0
paid_ratio=0.07831250000000001
```

False control:

```text
decision=FAIL_R488_A5C04_TERNARY_CONTROL_SYNDROME_GATE
candidate_mode=uncoded-repair-control
decode_error_rate=0.25
flip_tail_rate=1.0
correlated_tail_rate=1.0
correction_success_rate=0.8333333333333334
syndrome_lead=0.0
repair_loop_rate=0.25
candidate_ce=1.5123175869706462
paid_ratio=0.41781250000000014
```

Status:
`PASS_R488_A5C04_TERNARY_CONTROL_SYNDROME_GATE`.

Claim boundary:

R488 is a positive CPU finite ternary control-code gate for discrete control
families. It is not production v12 integration, open-ended instruction
semantics, a policy/refusal classifier, external judging, a repair loop,
training speed, runtime memory, or broad quality evidence.

---

## R489 R441 A9-C01 Packed Resident Tile Gate (2026-06-01)

Source: `R489` / `R441` / `A9-C01`

R489 implemented
`scripts/research/r441_a9_c01_packed_resident_tile_gate.py` and
`.workspace/training/rounds/r441_a9_c01_packed_resident_tile.ps1`. The gate
tests isolated direct packed ternary tile execution against a dense realized
reference and a dense-reconstruct false-control.

Diagnostic:

```text
decision=PASS_R489_A9C01_PACKED_RESIDENT_TILE_GATE
blocks=16
rows=64
cols=128
tile_cols=32
max_abs_error=1.9539925233402755e-14
relative_rmse=5.229277735517835e-16
dense_reference_bytes=262144
resident_packed_bytes=43008
packed_payload_bytes=32768
descriptor_bytes=8192
scale_bytes=2048
scratch_bytes=4096
dense_materialization_bytes=0
transient_dense_ratio=0.0
scratch_ratio=0.09523809523809523
resident_vram_ratio=0.1640625
hot_payload_load_events=0
dense_hot_bytes=524288
packed_hot_bytes=47104
projected_hot_speedup=11.130434782608695
claim_device=simulated_resident_payload_cpu_direct_execution
```

False control:

```text
decision=FAIL_R489_A9C01_PACKED_RESIDENT_TILE_GATE
candidate_mode=dense-reconstruct-control
dense_materialization_bytes=262144
transient_dense_ratio=6.095238095238095
scratch_ratio=6.190476190476191
hot_payload_load_events=1024
projected_hot_speedup=0.9175627240143369
```

Status:
`PASS_R489_A9C01_PACKED_RESIDENT_TILE_GATE`.

Claim boundary:

R489 is a positive isolated CPU direct packed-tile execution gate with exact
dense-reference agreement and receipt-level no-dense-materialization evidence.
It is not production v12 integration, a GPU kernel/runtime result, measured
tok/s, training speed, or broad quality evidence.
