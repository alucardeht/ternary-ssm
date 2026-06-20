# R11-A6 — Ternary Hardware Architecture Synthesis

**Round:** R11 | **Aspect:** A6 | **Waves:** W1 (ISA/Software) + W2 (Algebra/Semantics)
**Date:** 2026-03-06 | **Synthesis Agent:** Wave 3

---

## Summary Table

| ID | Finding | Expected Gain | Source | Verdict |
|----|---------|---------------|--------|---------|
| R11-A6-W1-A-F1 | Unified ISA: `PackedTernary` + dispatch | 1000× pack speedup, single ternary abstraction | W1-AgA | **IMPLEMENT** |
| R11-A6-W1-A-F2 | AVX2 wiring in `ternary_gemv()` | 2–4× CPU inference throughput | W1-AgA | **IMPLEMENT** |
| R11-A6-W1-A-F3 | Fast popcount LUT vs 32-iter loop | 3–6× CPU popcount path speedup | W1-AgA | **IMPLEMENT** |
| R11-A6-W1-A-F4 | CPU SSM recurrence twin (BF16 h-path) | 1.4–1.8× CPU recurrent step throughput | W1-AgA | **IMPLEMENT** |
| R11-A6-W1-A-F5 | Vectorized `pack_ternary_int2()` | 1000× pack time reduction per layer | W1-AgA | **IMPLEMENT** |
| R11-A6-W1-B-F1 | GEMV backend dispatch hook | 8× custom chip gains without model changes | W1-AgB | **IMPLEMENT** |
| R11-A6-W1-B-F2 | Fused x_proj+dt_proj GEMV | 3–6% e2e step time, ~392 KB saved | W1-AgB | **IMPLEMENT** |
| R11-A6-W1-B-F3 | `FrozenTernaryOp` unpack cache at eval | 4× speedup for frozen layers | W1-AgB | **IMPLEMENT** |
| R11-A6-W1-B-F4 | Tiled AP/OP matmul (wire existing analyzer) | 10–20% inference throughput | W1-AgB | **IMPLEMENT** |
| R11-A6-W1-B-F5 | Fused SSM recurrent step (single pass) | Eliminates 3 intermediate allocations | W1-AgB | **IMPLEMENT** |
| R11-A6-W2-A-F1 | Scatter-add for ternary GEMV (semiring) | 10–50× circuit compiler at training | W2-AgA | **IMPLEMENT** |
| R11-A6-W2-A-F2 | GF(3) factored ternary at both stages | 2–4× FLOP reduction when r ≤ N/2 | W2-AgA | **IMPLEMENT** |
| R11-A6-W2-A-F3 | Functor fusion: dt_proj via dt_rank fuse | 3–5% e2e throughput, eliminates 392 KB | W2-AgA | **IMPLEMENT** |
| R11-A6-W2-A-F4 | Committed weights → INT2 only (no float) | 35–45 MB VRAM freed (59% committed) | W2-AgA | **IMPLEMENT** |
| R11-A6-W2-A-F5 | Zero-skip gate for fast-decay SSM heads | 25–40% SSM FLOP reduction, bandwidth | W2-AgA | **IMPLEMENT** |
| R11-A6-W2-B-F1 | Precomputed A^t LUT (TRS monoid) | 1–3% step time, eliminates 8192 pow ops | W2-AgB | **IMPLEMENT** |
| R11-A6-W2-B-F2 | E-graph: CB diagonal for reset heads | 5–8% SSD scan FLOP reduction | W2-AgB | **IMPLEMENT** |
| R11-A6-W2-B-F3 | Matrix associativity: fuse RLEC weights | 1–2% for RLEC layers, 26 KB saved | W2-AgB | **IMPLEMENT** |
| R11-A6-W2-B-F4 | Elementwise kernel fusion (output gate) | 0.2% but compounds w/ torch.compile | W2-AgB | **IMPLEMENT** |
| R11-A6-W2-B-F5 | Cast hoisting in _ssd_scan_fla | 2–3 µs saved per FLA layer | W2-AgB | **IMPLEMENT** |

**Total: 20 findings, all passing Gate v3 (G1/G2/G3: actionable + ternary-compatible + always-active).**

---

## Detailed Findings

### ISA Abstraction Layer (W1-AgA: F1–F5)

**F1: R11-A6-W1-A-F1 — Unified ISA: `PackedTernary` + Backend Dispatch**

New unified `PackedTernary` struct with `backend_tag` field. Single `ternary_gemv_dispatch()` replaces 3 incompatible pack formats. Enables backend fallback (Triton → popcount → AVX2 → scalar) without code duplication.

- **File:** `src/model_v7/kernels/ternary_isa.py` (new) + `src/model_v7/inference.py` ~172
- **Pseudocode:** Unified struct with `w_val`, `w_sign`, `w_sum`, `scales`, `backend_tag` fields; dispatch table routes to `_DISPATCH_TABLE[tag](x, packed)`
- **Expected gain:** 1000× pack speedup (O(N*K) Python loop → O(1) tensor ops); inference dispatch ~80→10 lines; custom accelerator plug-in via `PackedTernary` contract only.
- **Verdict:** **IMPLEMENT**

**F2: R11-A6-W1-A-F2 — AVX2 Wiring Bug in `ternary_gemv()`**

Current code detects AVX2 via CPUID (lines 46–68) but entry point always calls scalar path. AVX2 twin unpacks 2-bit ternary weights 4 per byte into 32 INT8 activations per 256-bit load; replaces 128 scalar branches with vectorized `_mm256_or_si256`/`_mm256_sub_epi8`/`_mm256_add_epi32` sequence.

- **File:** `src/model_v7/kernels/c_portable/ternary_gemv.c` ~76
- **Pseudocode:** Replace entry point: `if (has_avx2()) ternary_gemv_avx2(...); else ternary_gemv_scalar(...);` with AVX2 inner loop using `_mm256_add_epi32` for ternary MACs.
- **Expected gain:** 2–4× CPU inference throughput at zero quality cost (AVX2 processes 32 elements/cycle vs 1 for scalar).
- **Verdict:** **IMPLEMENT**

**F3: R11-A6-W1-A-F3 — Fast Popcount LUT (vs 32-Iteration Loop)**

Current `popcount_sim()` uses 32-iteration bit-shift loop materializing `(N, 32)` temporary. Hardware `POPCNT`/`CNT` = 1 cycle. Software equivalent: 256-entry LUT on 4 uint8 nibbles replaces 32-wide temp with 4-element gather (8× fewer memory ops).

- **File:** `src/model_v7/kernels/popcount_gemv.py` ~35
- **Pseudocode:** `_POPCNT_LUT = torch.tensor([bin(i).count('1') for i in range(256)]); b = x.view(uint8); return _POPCNT_LUT[b.long()].reshape(..., 4).sum(-1)`
- **Expected gain:** 3–6× CPU popcount path speedup; eliminates `(N*M, 32)` → `(N*M, 4)` allocation (8× smaller).
- **Verdict:** **IMPLEMENT**

**F4: R11-A6-W1-A-F4 — CPU SSM Recurrence Twin (BF16 h-Bandwidth)**

Current fused Triton kernel (14 ops) requires CUDA; CPU falls back to 9+ separate ops. CPU twin using same interface: pre-cast `h` to BF16 for `dA*h` multiply (dA ∈ (0,1), BF16 sufficient) while keeping FP32 accumulation halves h-read bandwidth.

- **File:** `src/model_v7/kernels/triton/fused_ssm_recurrence.py` ~21 + `src/model_v7/inference.py` ~24
- **Pseudocode:** `dA = exp(dt*A); h = dA*h + dt*B*x` with `h` pre-cast to BF16 for decay multiply; all in PyTorch/ctypes, portable.
- **Expected gain:** 1.4–1.8× CPU recurrent step throughput; single function replacing 9+ launches; directly specifies custom accelerator SSM unit.
- **Verdict:** **IMPLEMENT**

**F5: R11-A6-W1-A-F5 — Vectorized `pack_ternary_int2()` (Replaces Python Loop)**

Current `pack_for_c_portable()` uses Python double-loop over `(N, K_padded)` — 2.3M iterations for 768×3072 matrix, taking seconds per layer. Three divergent pack paths exist. Unified vectorized primitive using bit ops: `codes = (w>0) | ((w<0)<<1)` then pack 4 trits/byte via `(a | b<<2 | c<<4 | d<<6)`.

- **File:** `src/model_v7/kernels/c_portable/wrapper.py` ~110–135 + `src/model_v7/kernels/pack_weights.py` ~36–44
- **Pseudocode:** `codes = (w>0).uint8() | ((w<0).uint8()<<1); c = F.pad(codes, ...).reshape(-1, 4); return (c[...,0] | c[...,1]<<2 | c[...,2]<<4 | c[...,3]<<6).uint8()`
- **Expected gain:** 1000× pack time reduction per layer (768×3072: ~2 s → ~2 ms); unifies 3 divergent encodings; maps to `TPACK` hardware instruction.
- **Verdict:** **IMPLEMENT**

---

### Hardware-Agnostic Interfaces (W1-AgB: F1–F5)

**F1: R11-A6-W1-B-F1 — GEMV Backend Dispatch Hook**

Replace hot-path `F.linear()` dispatch in all 4 projection layers with indirection through module-level `_gemv_backend` callable. Custom accelerator registers itself once; model code unchanged. Training path wraps with STE; eval path calls directly.

- **File:** `src/model_v7/components/ternary.py` ~2710 (TernaryLinear.forward)
- **Pseudocode:** `_GEMV_BACKEND = None; register_gemv_backend(fn); output = _GEMV_BACKEND(...) if _GEMV_BACKEND else F.linear(...)`
- **Expected gain:** 0% training overhead (branch-predicted); enables 8× custom chip bandwidth gains (885 KB on-chip SRAM) without model changes.
- **Verdict:** **IMPLEMENT**

**F2: R11-A6-W1-B-F2 — Fused x_proj+dt_proj GEMV (Related to W2-A-F3)**

SSM forward: x_proj produces `(B, T, dt_rank + 2*d_state)`, immediately consumed by dt_proj on `x_dbc[:,:,:dt_rank]` slice. Two sequential GEMV on same input. Fuse into single wider F.linear; custom chip handles both projections in on-chip pass. **Note:** W2-A-F3 approaches from functor (quantization commutes); this is pure dataflow fusion. Both complementary.

- **File:** `src/model_v7/components/ssm.py` ~1097–1102
- **Pseudocode:** `W_fused = cat([x_proj.weight; dt_proj.weight @ x_proj.weight[:dt_rank]]); x_dbc_dt = F.linear(x_conv, W_fused)`
- **Expected gain:** 3–6% e2e step time; eliminates 1 kernel launch + `(B, T, dt_rank)` allocation (~392 KB); on chip: single on-chip pass for both projections.
- **Verdict:** **IMPLEMENT**

**F3: R11-A6-W1-B-F3 — `FrozenTernaryOp` Unpack Cache at Eval**

Current `FrozenTernaryOp.forward()` calls `_unpack_ternary()` on every call — O(out_features * in_features) Python loop every step, zero caching. Cache unpacked weight on first call (or at `model.eval()`); packed INT2 form is canonical (hardware stays in SRAM); unpacking is software fallback detail.

- **File:** `src/model_v7/components/frozen_ternary_op.py` ~133
- **Pseudocode:** `if not hasattr(self, '_weight_cache'): w = _unpack_ternary(...); self._weight_cache = w.to(dtype); return F.linear(x, self._weight_cache, self.bias)`
- **Expected gain:** 4× speedup for frozen layers; eliminates ~590K Python iterations per call (768×768 layer); defines hardware boundary: packed form is canonical.
- **Verdict:** **IMPLEMENT**

**F4: R11-A6-W1-B-F4 — Tiled AP/OP Matmul (Wire Existing Analyzer)**

Existing `adaptive_dataflow.py` DataflowSelector correctly analyzes layer dataflow (AP vs OP) and computes `tile_k` per layer at init, but result never used: `TernaryLinear.forward()` always calls `F.linear()`. Wire analyzer output into execution: replace `F.linear()` with `ap_matvec()` or `op_matvec()` selected per-layer, respecting L1 cache capacity (tile_k=64 → 128 bytes FP16 in registers).

- **File:** `src/model_v7/components/ternary.py` ~2713 + `src/model_v7/components/adaptive_dataflow.py` ~102
- **Pseudocode:** `self._tile_k = dataflow_selector.get_tile_params(name)['tile_k']; if self._dataflow=='AP': output = ap_matvec(x, w, tile_k=self._tile_k)`
- **Expected gain:** 10–20% inference throughput on CPU (cache-resident tiles); custom chip: scratchpad buffer allocation = tile_k.
- **Verdict:** **IMPLEMENT**

**F5: R11-A6-W1-B-F5 — SSM Recurrent Step Fused (x_proj+dt_proj+state+output)**

At inference (seq_len=1), recurrent mode: 4 separate kernel launches (x_proj GEMV, dt_proj GEMV, h update, output). Fuse into single `ternary_ssm_step(x_conv, h, ssm_module)` function: stream x_conv once, accumulate B/C/dt in registers, compute h_update inline, produce y without DRAM writes for intermediates. Hardware primitive contract: (activation, state, weight pointers) → (y, h_new).

- **File:** `src/model_v7/components/ssm.py` ~879 (recurrent branch)
- **Pseudocode:** Single function: `x_dbc=x_proj(x_conv); dt=F.softplus(dt_proj(...)); B,C extracted; h_new=A_bar*h+B*x; y=C*h+D*x` — all intermediate B/C/dt in registers, no tensor allocation.
- **Expected gain:** Eliminates 3 intermediate tensor allocations per step (d_inner + dt_rank + d_state = 256 floats ≈ 512 bytes); on chip: optimal memory access (input once, weights once, state read/write, output once).
- **Verdict:** **IMPLEMENT**

---

### Algebraic Semantics (W2-AgA: F1–F5)

**F1: R11-A6-W2-A-F1 — Scatter-Add for Ternary GEMV (Semiring Zero-Annihilation)**

Semiring property: `w ∈ {-1,0,+1} ⟹ w*x ∈ {-x, 0, +x}`. Current `TernaryCircuitCompiler.execute()` uses Python loop over nnz entries; replace with `torch.scatter_add_()` on precomputed pos/neg index tensors (2 kernel calls regardless of nnz). Enables circuit compiler at training time for committed layers.

- **File:** `src/model_v7/components/ternary_circuit_compiler.py` ~120
- **Pseudocode:** Precompute pos_rows, pos_cols, neg_rows, neg_cols from CSR at compile time; in execute: `y.scatter_add_(1, pos_row_t, x_flat[:,pos_col_t]); y.scatter_add_(1, neg_row_t, -x_flat[:,neg_col_t])`
- **Expected gain:** 10–50× CPU throughput vs loop (59% sparsity: nnz = 0.59*N*K); enables training-time use of circuit compiler; hardware: scatter-add unit (adder tree + mux) replaces multiplier array.
- **Verdict:** **IMPLEMENT**

**F2: R11-A6-W2-A-F2 — GF(3) Factored Ternary at Both Stages**

GF(3) Gaussian elimination: W = U @ V, both factors provably ternary (field operations preserve ternary). Current `factored_matvec()` uses two F.linear calls on FLOAT U, V, discarding ternary structure. Both stages use scatter-add only (no multiplier); if r = N/4 (plausible for ternary matrices), total FLOPs drop from N*K to (N+K)/4 (4× reduction).

- **File:** `src/model_v7/components/gf3_factorization.py` ~191
- **Pseudocode:** `plan_v = compiler.compile([V]); v_out = compiler.execute(x); plan_u = compiler.compile([U]); return compiler.execute(v_out)` — both stages use scatter-add.
- **Expected gain:** 2–4× FLOP reduction when r ≤ N/2; hardware: two adder arrays (width r and N), no multiplier unit needed for factored path.
- **Verdict:** **IMPLEMENT**

**F3: R11-A6-W2-A-F3 — Functor Fusion: dt_proj via Quantization Commutes**

Quantization functor Q (sign-threshold) approximately commutes with composition when intermediates are sign-dominated: `Q(W_dt @ Q(W_x[:dt_rank])) ≈ Q(W_dt @ W_x[:dt_rank])`. Fused weight `W_fused = Q(W_dt @ W_x[:dt_rank])` computable at eval prep, single ternary GEMV replaces both projections for dt output. **Related to W1-B-F2 (dataflow fusion); different scope — quantization-aware vs pure concatenation.**

- **File:** `src/model_v7/components/ssm.py` ~1097–1102
- **Pseudocode:** At model.eval(): `W_x = lsq_ternary(x_proj.weight); W_dt = lsq_ternary(dt_proj.weight); W_dt_fused = Q(W_dt @ W_x[:dt_rank]); x_bc = F.linear(x_conv, W_x[dt_rank:]); dt = ternary_gemv(x_conv, W_dt_fused_t)`
- **Expected gain:** 3–5% e2e throughput; eliminates `(B, T, dt_rank)` intermediate (~392 KB); custom chip: dt_proj SRAM bank replaced by W_dt_fused_t, single on-chip pass.
- **Verdict:** **IMPLEMENT**

**F4: R11-A6-W2-A-F4 — Committed Weights → INT2 Only (Adjunction Unit/Counit)**

Adjunction between Enc (quantize) and Dec (scale-multiply): for committed groups, packed INT2 is the FREE construction (unique minimal faithful); float shadow is COUNIT (redundant round-trip). Once committed_mask[g]=True, zero out float shadow, read only from INT2 buffer. Current code reads float shadow every step even for committed groups (mathematically unnecessary by adjunction identity).

- **File:** `src/model_v7/components/ternary.py` ~2325 + `src/model_v7/components/autofreeze.py` (commit event)
- **Pseudocode:** At commit: `w_trit = quantize_ternary(weight[g*gs:(g+1)*gs]); self._committed_int2[g] = pack_ternary_int2(w_trit); self.weight.data[g*gs:(g+1)*gs].zero_()`; in forward: read from `_committed_int2` for committed groups.
- **Expected gain:** 35–45 MB VRAM freed (59% committed @ 6M model); eliminates float read+quantize per step for committed; INT2 packing: 16 bytes vs 256 bytes FP32 per group_size=128; zero quantize overhead for committed path.
- **Verdict:** **IMPLEMENT**

**F5: R11-A6-W2-A-F5 — Zero-Skip Gate for Fast-Decay SSM Heads (Monoid Absorbing Element)**

SSM recurrence: h_new = A_bar*h + B*x. When A_bar < epsilon, the monoid element A_bar is absorbing (L_0 = L_0 ∘ anything), so h_new = B*x exactly (history forgotten). Hardware gate: comparator per head suppresses h SRAM read when A_bar < eps; software: `torch.where(fast_mask, zeros, dA*h)`. Empirically 25–40% of heads are fast-decay.

- **File:** `src/model_v7/components/ssm.py` ~1238
- **Pseudocode:** `dA = exp(dt*A); fast_mask = (dA < 0.01); h_new = torch.where(fast_mask, torch.zeros_like(h), dA*h); h_new = h_new + dt*B*x`
- **Expected gain:** 25–40% SSM recurrent FLOP reduction; hardware: comparator gate reduces h-state bus bandwidth 25–40%; CPU-portable (pure torch.where); always-active.
- **Verdict:** **IMPLEMENT**

---

### Computation Graph Rewriting (W2-AgB: F1–F5)

**F1: R11-A6-W2-B-F1 — Precomputed A^t LUT (TRS Monoid Homomorphism)**

In `_lti_fft_conv()`, A^t is recomputed for each t ∈ [0, seq_len) every step. A is static (not input-dependent); map t → A^t is monoid homomorphism (Z_≥0, +) → scalars. Precompute table of length max_seq_len=512 once at init, slice `[:seq_len]` each forward — reduces O(n_heads × seq_len) pow ops to single tensor slice. TRS rule is confluent: one normal form.

- **File:** `src/model_v7/components/ssm.py` ~862
- **Pseudocode:** At __init__: `t_table = arange(max_seq_len); self.register_buffer('_A_pow_table', A_val.unsqueeze(-1).pow(t_table))`; in forward: `A_powers = self._A_pow_table[:, :seq_len]` (no pow call).
- **Expected gain:** 1–3% step time reduction; eliminates 8192 pow ops per step per layer (16 heads × 512 seq); also eliminates `t = arange(...)` allocation per step.
- **Verdict:** **IMPLEMENT**

**F2: R11-A6-W2-B-F2 — E-graph: CB Diagonal for Reset Heads (Zero-Annihilator)**

In `_ssd_scan()`, CB einsum produces `(B, T, T)` matrix at O(B × T² × d_state) cost. When A_head = 0 (reset mode, ternary), decay L[t,s] = 1 but recurrence has no memory: h_t = B*x_t only. CB cross-product contributes only diagonal `C[t]·B[t]` (s=t); off-diagonal terms zeroed by scan. E-graph rewrite replaces full O(T²) einsum with O(T) diagonal for reset-mode heads (static `A_ternary==0` mask).

- **File:** `src/model_v7/components/ssm.py` ~1209
- **Pseudocode:** `reset_heads = (A==0).nonzero(...)[0]; CB_diag = (C[:,:,reset_heads] * B[:,:,reset_heads]).sum(-1)` (O(T)); active_heads use full einsum.
- **Expected gain:** 5–8% SSD scan FLOP reduction if ~1/3 heads have A=0; memory for CB drops from `(B,T,T,n_heads)` to `(B,T,n_heads)` for reset partition.
- **Verdict:** **IMPLEMENT**

**F3: R11-A6-W2-B-F3 — Matrix Associativity: Fuse RLEC Weights (TRS Normal Form)**

RLEC correction in `TernaryLinear.forward()` applies `A_corr(B_corr(x))` — two sequential linear maps. By associativity, `(A_corr @ B_corr) @ x` is the right-associated normal form: single GEMM, no intermediate. Rewrite applicable at model.eval() entry (r < min(in,out)/2 always true for r=8 on 768×768 layer); confluent (matrix mult associativity). Transparent to STE (correction weights FP32-only, not quantized).

- **File:** `src/model_v7/components/ternary.py` ~2769
- **Pseudocode:** At model.eval(): `W_rlec = (self.correction_A @ self.correction_B).detach(); self.register_buffer('_rlec_fused', W_rlec)`; in forward: `output = output + F.linear(x, self._rlec_fused.to(dtype))`.
- **Expected gain:** 1–2% for RLEC-enabled layers; eliminates 1 GEMM kernel launch + `(B, T, r)` intermediate (~26 KB at BF16 for r=8); at 5 µs per kernel launch.
- **Verdict:** **IMPLEMENT**

**F4: R11-A6-W2-B-F4 — Elementwise Kernel Fusion (Output Gate Distributivity)**

In SSM output computation (lines 1174–1178), 4 sequential elementwise ops: `y_skip = x_conv*D`, `y_total = y+y_skip`, `act_z = act_fn(z)`, `output = act_z*y_total`. By distributivity `a*(b+c) = a*b+a*c`, fuse to single elementwise kernel: `output = act_fn(z)*(y + x_conv*D)`. PyTorch fuses 3 elementwise ops into 1 kernel via TorchScript/compile. E-graph rule: one lower-cost representative in the E-class.

- **File:** `src/model_v7/components/ssm.py` ~1174 (both SelectiveSSM and SelectiveSSMSSD)
- **Pseudocode:** Replace 4 ops with: `D_view = self.D.view(1,1,-1); output = self.out_proj(self.act_fn(z) * (y + x_conv*D_view))`
- **Expected gain:** 0.2% base (~15 µs saved per SSM layer) but compounds with torch.compile; saves 2 intermediate allocations `(4, 512, 192)` ≈ 3 MB.
- **Verdict:** **IMPLEMENT**

**F5: R11-A6-W2-B-F5 — Cast Hoisting in _ssd_scan_fla (Graph Grammar)**

In `_ssd_scan_fla()`, 5 separate BF16→FP32 casts on derived tensors: dt_heads, g, q, k, v. Graph grammar "cast hoisting" rule: if all consumers require FP32, hoist cast to source (or single boundary cast). Insert `dt_f32 = dt.float()` at entry; all downstream inherit FP32 (idempotent: cast(cast(x)) = cast(x)). Eliminates 4 redundant full-tensor casts on `(B,T,n_heads,d)` tensors.

- **File:** `src/model_v7/components/ssm.py` ~1238 (_ssd_scan_fla)
- **Pseudocode:** `dt_f32 = dt.float()` (one cast at entry); derive dt_heads_f32, g, k, v downstream — all inherit fp32 without per-tensor casts.
- **Expected gain:** 2–3 µs saved per FLA layer (2–3 MB memory traffic eliminated); reduces `torch.amp.autocast` context manager overhead.
- **Verdict:** **IMPLEMENT**

---

## Cross-Wave Complementarities and Dedup Notes

- **W1-B-F2 vs W2-A-F3:** Both fuse x_proj+dt_proj. W1-B-F2 is pure dataflow (concatenate weights for single F.linear); W2-A-F3 is quantization-aware (Q commutes, dt path alone). Both applicable — different optimization scopes.
- **W1-B-F4 (tiled AP/OP) vs W1-B-F1 (dispatch hook):** Complementary. F1 enables custom chip backend; F4 wires existing analyzer into execution for CPU/GPU tiling.
- **W2-A-F1 (scatter-add) vs W2-A-F2 (GF(3) ternary stages):** F1 is scatter-add for any sparsity; F2 specifically applies scatter-add to both GF(3) factors.

**No duplicates detected across W1-W2. All 20 findings are distinct and complementary.**

---

## Implementation Checkpoint

- **All findings pass Gate v3:** Actionable (specific files + line ranges), compatible with ternary {-1,0,+1}, always-active (no OFF-by-default flags).
- **Ordered by expected gain:** F1 (1000×) and F5 (1000×) pack speedups first; then CPU inference (F2, F3, F4); then algebraic simplifications (W2-A, W2-B).
- **Hardware target:** Every finding specifies its custom accelerator equivalent (scatter-add unit, on-chip SRAM patterns, dispatch contracts, gate signals).

Next phase: architecture-enforcer validates recipes; code-writer implements sequentially.
