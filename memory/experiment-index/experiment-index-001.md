# Experiment Index

**Last updated:** 2026-04-28

## v10 Experiments (R24-R121) — src/model_v10/experiments/

| # | Name | Result | Status |
|---|------|--------|--------|
| R24 | V9 compilation pipeline + byte-level OLS | Byte OLS +3.15% vs random; vocab=50K OLS fails (0.65 obs/token); learning without GD confirmed at vocab=256 | CONFIRMED |
| R52 | FP layer-sequential training N=4 | Baseline: val=5.765 nats, optimizer VRAM O(1) per stage | CONFIRMED |
| R55 | FP scaling N=16 | val=6.187 (+0.422 vs N=4); penalty saturation begins | CONFIRMED |
| R60 | FP scaling N=32 — NaN investigation | NaN at step ~200 stage 8; root cause: residual amplification through 31 random frozen layers | CONFIRMED |
| R61 | FP N=32 small batch (near-zero init fix) | val=6.201 (+0.436 vs N=4); near-zero out_proj init for future layers eliminates NaN | CONFIRMED |
| R62 | FP N=32 large batch (B=8, T=384) | val=5.952 (+0.187 vs N=4); large batch reduces FP penalty 57%; 32K steps zero NaN | CONFIRMED |
| R63 | N=64 FP scaling — pending | Not executed | DESIGNED |
| R71 | ALS on Mamba-2 SSD | ALS diverges on SSD (non-linear); guaranteed only for MSE not CE | REJECTED |
| R74 | Mathematical analysis: why GD is structurally required in Mamba-2 SSD | 3 independent reasons (W_x quadratic, temporal credit non-locality, gate coupling); SSMDP proposed as GD-free alternative | THEORETICAL |
| R75 | SSMDP OLS pure proof-of-concept | FAILED: cond(H^T H)=1.4e10; random B,C → OLS≈zero (expected result) | REJECTED |
| R76 | SSMDP OLS mid-training ("compile first fine-tune second") | Neutral/slightly harmful: +0.085 nats (d=256), +0.018 nats (d=512); OLS on frozen H becomes stale | NEUTRAL |
| R77 | SSMDP+GD vs SSD+GD quality gap | SSMDP viable: gap <0.5 nats at both scales (d=256: +0.402, d=512: +0.457) | CONFIRMED |
| R78 | Pure linear SSM + full ALS (B, C, W_out) | FAILED: ALS diverges step 3, cond explodes 226→796K; CE loss breaks ALS convergence guarantee | REJECTED |
| R79 | HiPPO A init + OLS W_out vs random init | HiPPO+OLS: 7.63 vs random 8.34 nats at 100 steps; 2-3× data efficiency gain | CONFIRMED |
| R80 | Extended convergence HiPPO+OLS vs random (200 steps) | Gap shrinking: 0.71 nats at step 100 → 0.31 nats at step 200; gap → 0 at ~300-400 steps | CONFIRMED |
| R82 | FP vs Full GD data efficiency N=16 | Full GD -0.32 nats better per token; FP_LARGE (8× tokens) worse than FP_SMALL | CONFIRMED |
| R83 | FP vs Full GD N=32 VRAM proxy test | Full GD -0.45 nats better; FP quality cost GROWS with depth at insufficient budget | CONFIRMED |
| R84 | FP-warmup + Full GD hybrid | Hybrid +0.38 nats WORSE than pure GD; basin mismatch confirmed | REJECTED |
| R85 | FP vs GD at convergence (200 steps/stage, N=8) | Gap shrinks to 0.18 nats (vs 0.32-0.45 at 25 steps); structural gap persists | CONFIRMED |
| R86 | FP quality gap vs d_model scaling | Gap ~7% (0.145-0.156 nats) structurally stable across d=256,512,1024; scale-invariant | CONFIRMED |
| R87 | FP quality gap vs n_layer scaling (N=8→16, 200 steps) | FP gap CLOSES with depth: 0.152→0.084 nats (45% reduction); FP scales better than GD at N=16 | CONFIRMED |
| R88 | FP gap N=32 validation (50 steps/stage) | Gap widens at insufficient budget; R87 trend requires 200+ steps/stage to hold | CONFIRMED |
| R89/R89b | FP vs GD N=32, 200 steps/stage | Non-monotonic: N=16 sweet spot (gap=0.084), N=32 gap=0.158 (regresses from N=16) | CONFIRMED |
| R90 | FP + INT2 layer storage VRAM analysis | 7B projected VRAM = 6.16 GB with INT2 inactive layers; fits 12 GB | CONFIRMED |
| R91 | Backward CE coordinate descent | Gap closed: 0.9% (negligible); inter-layer catastrophic interference at k=3 | REJECTED |
| R92 | Multi-cycle forward BCD (2 cycles) | BCD Cycle 2: 32.4% gap closure; Cycle 3 catastrophic divergence at LR=1e-6 | CONFIRMED |
| R93 | BCD LR test — 2 cycles same LR (5e-6) | Cycle 2: -12.7%; Cycle 3: +28.0%; net ~28-32% gap closure; oscillation is structural not LR | CONFIRMED |
| R94 | Reservoir computing diagnostic — 1-layer random | 78.1% of FP quality is structural (free from random init + lm_head training) | CONFIRMED |
| R94b | Reservoir computing diagnostic — 8-layer random | 72.8% structural (slightly worse than 1-layer); multi-layer reservoir not better | CONFIRMED |
| R96 | PGLR selective BCD top-3 layers | Worsened FP by +0.006 nats; design flaw (GD=200 steps as warm-up biases result) | REJECTED |
| R97 | FP + ternary training vs FP + fp32 | Ternary training overhead: +1.06 nats; use INT2 storage not ternary training | REJECTED |
| R98 | INT2 storage for frozen layers during FP | Delta: +0.025 nats (below noise floor); INT2 storage VIABLE for frozen layers | CONFIRMED |
| R99 | Scale test N=16, d=512 | Gap = +0.1085 nats; INT2 overhead ≈ -0.035 (negligible); gap SMALLER than N=8 | CONFIRMED |
| R100 | Real INT2 packing (del+uint8) | 244 MB → 41 MB (83% compression); 7B = 4.53 GB; 16× per layer | CONFIRMED |
| R101 | Scale test N=32 | Gap = 0.1545 (LARGER than N=16=0.11); gap is non-monotone with scale | CONFIRMED |
| R102 | Real data confirmation WikiText-103 | INT2 overhead +0.019 nats; 5th data point confirming INT2 viability | CONFIRMED |
| R108/R108b | QOTA (quadratic offset ternary adaptation) | +1.20 nats worse; Taylor expansion invalid for delta=O(1) | REJECTED |
| R109 | TRGL (+0.016) and DGL (Deep Gradient Lifting, +0.169) | TRGL neutral; DGL prejudicial | REJECTED |
| R110 | DEQ-FP (deep equilibrium for FP gap) | NOT VIABLE: distance=0.84, L=0.97 (requires L<0.5, distance<0.3) | REJECTED |
| R112 | Reverse BCD | -59% gap (WORSE); violates residual stream asymmetry | REJECTED |
| R113 | SLS (Surrogate Layer Stack) N=50 | +14.8% gap closure; surrogates too weak when val>2.5 | CONFIRMED |
| R114 | SLS N=100 + BCD | +23.2% gap closure; best combo at the time | CONFIRMED |
| R115 | Warm Init alpha=0.3 + SLS + BCD | Warm Init alone: +12%; All Combined: -33.4% (systemic mods interfere) | CONFIRMED |
| R116 | FP H_k caching (fp32) | 2.59× speedup (16.3s→6.3s); quality +0.046 nats better; 7B cache=16-32 GB (too big) | CONFIRMED |
| R117 | CPU+GPU hybrid forward | FP32 hybrid: -0.028 nats better; FP16: +0.010 nats; INT2 forward: +0.067 nats (rejected) | CONFIRMED |
| R118 | Activation quantization sweep | INT8 cache: -0.028 nats (BEST), FP16: -0.038, INT4: +0.302 (rejected), Ternary: +0.180 (viable at 16×) | CONFIRMED |
| R119 | Integrated pipeline (INT8 cache + ternary act) | INT8 cache 2.28× compression; ternary acts 2GB@7B; all fit 12GB | CONFIRMED |
| R120 | H_k distribution diagnostic | H_k is LOW-RANK: stable_rank=2-5, rank@90%=17-20/256 | CONFIRMED |
| R120b | PCA-INT8 advanced cache compression | 51× compression (50MB→1MB at N=8); quality +0.114 nats vs fp32 | CONFIRMED |
| R121 | 7B proof-of-concept (3.38B params) | VRAM stage 0: 1.33 GB (4.6× better than R90 projection); loss 5.71→3.39 in 150 steps | CONFIRMED |

## v11 Experiments (R122-R296) — src/model_v11/experiments/

| # | Name | Result | Status |
|---|------|--------|--------|
| R122 | Cached forward propagation (fp32 vs PCA-INT8) | fp32 cache: +0.0176 nats, 2.59× speedup, ~10GB@7B; PCA-INT8: catastrophic (+3.04 nats) — error compounds exponentially for training cache | CONFIRMED |
| R122b | Bottleneck architecture analysis (factored projections) | All variants rejected: narrow residual loses quality, factored reduces to 269M params | REJECTED |
| R123 | K-Layer Chunk FP (K=1,2,4) | K=4: -0.106 nats vs K=1 (significant); K=2≈K=1 (noise); phase transition between K=2 and K=4 | CONFIRMED |
| R123b | K-Layer Chunk FP scale test (d=512, N=16) | K=4 still best (-0.025 nats) but improvement SHRINKS with scale (0.106→0.025); advantage may vanish at 7B | CONFIRMED |
| R124 | Theoretical research: closing FP structural gap | M7 (Adjoint State Method) identified as most promising; P4 (LR-ATP) strongest paradigm alternative | THEORETICAL |
| R125 | Adjoint rank diagnostic | lambda_k IS low-rank: stable_rank=3.9-6.5, rank@90%=19-37 (<15% of d=256); M7 prerequisite VALIDATED | CONFIRMED |
| R126 | Adjoint FP correction smoke test | 20.9% gap closure with 10 correction steps (0.8% compute overhead); LR=1e-4 worsens, LR=1e-5 helps | CONFIRMED |
| R127 | CTII init + extended correction (50 steps) | CTII (orthogonal init): +0.072 nats worse; 50 correction steps overshoots (-250%); M3 DEAD | REJECTED |
| R128 | Sequential adjoint correction (SAC) | 35.1% gap closure with 3 sweeps × 8 layers = 24 steps; best post-FP correction method | CONFIRMED |
| R129 | Pipeline integration (FP + INT2 + PCA-INT8 + SAC) | Full pipeline runs end-to-end; VRAM peak 893 MB; ~50s per stage; 29 min total | CONFIRMED |
| R130 | Cached 7B pipeline end-to-end (3.38B params) | O(n) linear scaling achieved (43× vs uncached); 29.4 min total; 893 MB peak VRAM; val 5.72→4.54 | CONFIRMED |
| R132 | Compilation training (torch.compile) | File exists; no log entry found in implementation log | DESIGNED |
| R133a | Stable rank at 7B scale | File exists; no detailed log entry found | DESIGNED |
| R133b | K-FAC vs Adam | File exists; no detailed log entry found | DESIGNED |
| R133c | TCC/SCC validation | File exists; no detailed log entry found | DESIGNED |
| R134 | Scale gap measurement | File exists; no detailed log entry found | DESIGNED |
| R135 | FP refinement (SBP post-FP, 89% gap closure claimed) | R140 references: 89% gap closure in ~1000 SBP steps at 6M scale | CONFIRMED |
| R136 | LSGD group-4 | Dead: passes ZERO gradient through frozen boundaries (100% error); rejected | REJECTED |
| R137a | SGD vs Adam comparison | File exists; no detailed log entry found | DESIGNED |
| R137b | Distribution matching | File exists; no detailed log entry found | DESIGNED |
| R137c | Gradient profile | File exists; no detailed log entry found | DESIGNED |
| R138 | Orthogonal FP (frequency-segregated A_log) | File exists; no detailed log entry found | DESIGNED |
| R138b | Orthogonal FP + shared lm_head | No reproducible effect: Run1 -0.016 (marginal better), Run2 +0.100 (worse); variance dominates | NEUTRAL |
| R139 | Sequential Backprop (SBP) | SBP quality ≈ joint GD (Δ=+0.0002 nats); VRAM 2.4× less (293 MB vs 695 MB); 5.5× slower | CONFIRMED |
| R140 | Speed optimization discussion round 1 | GPU utilization 14%; engineering table stakes identified; sign-gradient backward and LRJP proposed | THEORETICAL |
| R141 | Mathematical validation of R140 findings | F1 sign-grad: PARTIAL; F2 LRJP: PARTIAL (error accumulates geometrically); F5 collider mask: INVALID; F4 token budget: CONDITIONAL | CONFIRMED |
| R142 | SBP convergence curve | 65.1% gap closure at 100 SBP steps from FP checkpoint; accelerating (not asymptotic); 104 min total 7B projected | CONFIRMED |
| R143 | Engineering table stakes (batch, bf16, compile) | Batch=16: 4.1× speedup (winner); bf16: -18% (SLOWER at 6M); torch.compile: -32% (SLOWER at 6M) | CONFIRMED |
| R144 | Mathematical speed optimization research | OLS 1-shot for W_out (50×); DTA direct ternary assignment (20×); BIC backward info cascade (44-220×) identified | THEORETICAL |
| R144b | OLS + DTA implementation test | File exists; no detailed log entry found | DESIGNED |
| R145 | Mathematical validation of R144 findings | OLS: PARTIALLY VALID; DTA: PARTIALLY VALID; K-FAC: INVALID (residual destroys block-diagonal Fisher); GaLore: INVALID for ternary | CONFIRMED |
| R146 | FP gap research round 1 — information-theoretic analysis | FP gap = information deficit: delta_k available in 1 backward pass; OLS 1-shot formula derived; 86% survives ternary (medium confidence) | THEORETICAL |
| R147 | Gradient diagnostic + SSM dormancy discovery | out_proj dominates (0.74); x_proj tiny (0.004, 185× smaller); b_dt collapses → dt≈0 → SSM DORMANT; gradient chain dies | CONFIRMED |
| R148 | SSM dormancy mechanism: energy basin vs gradient shadow | CE_{t+1} fix KILLED: dormancy at b_dt=-inf gradient=0; reparametrization cannot fix information-theoretic problem | REJECTED |
| R149 | Reparametrization analysis: dt_min protection | KILLED: dt_min·exp(b_dt) = identical gradient to softplus at small dt (calculus identity); problem is incentive not coordinates | REJECTED |
| R150 | Training speed research round 4: acceleration strategies | Lazy cache refresh 2-3×; other proposals killed; FP acceleration bounded by O(n²) backward | THEORETICAL |
| R151 | Paradigm shift research round 5: radical approaches | Best stack: INT2 offload (2×) + rank reorg N=8/d=8192 (4×) + lazy cache (2×) = ~19× combined; rank diagnostic critical | THEORETICAL |
| R152 | Rank diagnostic (file exists: r152_rank_diagnostic.py) | File exists; no detailed log entry found | DESIGNED |
| R153 | Integer training research | INT16 clipped-sum: VALID; A discrete + dt fp16: VALID; integer margin loss: VALID; SPSA: KILLED | THEORETICAL |
| R153b | Sign-gradient depth theoretical validation | Pure sign training KILLED for depth >4: SNR=2% at k=31 (coin flip); DFA actually better than propagated signs below layer 16 | CONFIRMED |
| R154 | Bottom layer training research + validation | SNR(k)=2/sqrt(1+rho^2*k) confirmed; block-wise backprop k=4 mathematically mandated (k_max≈3.95) | CONFIRMED |
| R155 | Precision map + integer component validation | INT32 matmul (33-35% faster); BitShiftNorm (±0.02 nats); exp(A·dt) LUT (<0.01 nats); margin loss (0.01-0.03 nats worse) — all implemented | CONFIRMED |
| R155b | Empirical test battery for precision components | Pending / deferred | DESIGNED |
| R156 | SBP speed research + validation | INT8 boundary cache: 1.5-2× speedup (VALIDATED); low-rank J_k backward: KILLED (J_k full-rank); Adafactor: 3-5× if converges | THEORETICAL |
| R157 | Eliminate fp32 from backward + validation | 92% backward integer claim: VALID but misleading (memory only, not optimizer); TSA sign accumulator: 8× memory (untested); SPSA: KILLED | THEORETICAL |
| R158 | TSA vs Adam convergence test | TSA OUTPERFORMS Adam: -1.37 nats (33% better); 24% VRAM reduction (8.2 vs 10.8 GB); 5% faster; paradigm realization — TSA naturally aligns with discrete weights | CONFIRMED |
| R159 | Optimize joint-GD-TSA pipeline | Gradient checkpointing: 2GB→0.35GB activations; TSA threshold annealing: -0.018 nats (smoother); INT4 deferred; zero-memory accumulator KILLED | CONFIRMED |
| R160 | VRAM efficiency research | Low-rank accumulator: CONDITIONAL (late-stage only); sparse accumulator: UNPROVEN; CPU offload: VALIDATED ✓ (70% GPU VRAM reduction, 10-26% overhead) | THEORETICAL |
| R160a | Accumulator rank diagnostic (n=4096) | rank(A_k)=36-61 at d=4096 (NOT low-rank); PCA-rank-8 loses +4.2 nats at step 50; low-rank accumulator KILLED | REJECTED |
| R161 | Training speed research + validation | Batch=4: 1.5× (VALIDATED); K=8 parallel docs: 4-6× (VALIDATED); Pipeline 1F1B: complex; backward skip: deferred | THEORETICAL |
| R162 | VRAM v2 research + validation | CPU offload validated: 9.6→3.04 GB GPU; 10% overhead; separable accumulator mathematically false | CONFIRMED |
| R163 | Speed v2 research + profiler recommendation | torch.compile: 1.2× realistic (not 1.5-1.8×); profiler mandatory before any optimization | THEORETICAL |
| R163b | 7B reality check: all prior estimates were wrong | 9.3 tok/s measured (vs 850 at 6M); GPU utilization 0.006%; 74,617 days to train 60B tokens; d² scaling dominates | CONFIRMED |
| R164 | 7B VRAM research + validation | SBP per-stage A_k = 220 MB (not 7 GB); CPU offload unnecessary for SBP; training compute-bound not VRAM-bound | CONFIRMED |
| R165 | 7B speed research + validation | Save L in context: 30.6% speedup (IMPLEMENT); Truncated BPTT: plausible 1.5-2×; SPSA: KILLED; in_proj at 0.27% GPU utilization | CONFIRMED |
| R166 | VRAM 7B v3 research | File referenced; detailed analysis of backward VRAM decomposition | THEORETICAL |
| R170 | Groups test | File exists; no detailed log entry found | DESIGNED |
| R174 | 7B benchmark | File exists; no detailed log entry found | DESIGNED |
| R175 | Parameter count | File exists; no detailed log entry found | DESIGNED |
| R197 | OLS validation | File exists; no detailed log entry found | DESIGNED |
| R199 | TFLOPs benchmark | File exists; no detailed log entry found | DESIGNED |
| R200 | Combined benchmark | File exists; no detailed log entry found | DESIGNED |
| R209 | Model import test | File exists; no detailed log entry found | DESIGNED |
| R216 | Import test | File exists; no detailed log entry found | DESIGNED |
| R219 | SVD import | File exists; no detailed log entry found | DESIGNED |
| R219b | SVD validation | File exists; no detailed log entry found | DESIGNED |
| R227 | BGKV test | File exists; no detailed log entry found | DESIGNED |
| R228 | Full system test | File exists; no detailed log entry found | DESIGNED |
| R228b | Diagnostic | File exists; no detailed log entry found | DESIGNED |
| R228c | Long seq test | File exists; no detailed log entry found | DESIGNED |
| R228d | Window sweep | File exists; no detailed log entry found | DESIGNED |
| R229 | Ternary GPT-2 | File exists; no detailed log entry found | DESIGNED |
| R229b | Ternary clean | File exists; no detailed log entry found | DESIGNED |
| R229c | Calibrated ternary | File exists; no detailed log entry found | DESIGNED |
| R229d | Fast calibrated | File exists; no detailed log entry found | DESIGNED |
| R229e | GPTQ ternary | File exists; no detailed log entry found | DESIGNED |
| R230 | GPT-2 conversion | File exists; no detailed log entry found | DESIGNED |
| R234 | BGKV Dirac test | File exists; no detailed log entry found | DESIGNED |
| R235 | TopM herding test | File exists; no detailed log entry found | DESIGNED |
| R235b | Heuristic scoring | File exists; no detailed log entry found | DESIGNED |
| R236 | Hamming kernel | File exists; no detailed log entry found | DESIGNED |
| R237 | Long seq comparison | File exists; no detailed log entry found | DESIGNED |
| R241 | GPTQ ternary | File exists; no detailed log entry found | DESIGNED |
| R242 | Dual memory test | File exists; no detailed log entry found | DESIGNED |
| R243 | INT4 to ternary | File exists; no detailed log entry found | DESIGNED |
| R243b | GPTQ INT4 STE | File exists; no detailed log entry found | DESIGNED |
| R245 | 7B convergence | File exists; no detailed log entry found | DESIGNED |
| R246 | OLS W_out test | File exists; no detailed log entry found | DESIGNED |
| R247 | 7B FP 8 stages | File exists; no detailed log entry found | DESIGNED |
| R248 | 7B full train | File exists; no detailed log entry found | DESIGNED |
| R252 | K-FAC ITEE NUSB | File exists; no detailed log entry found | DESIGNED |
| R252b | K-FAC 7B test | File exists; no detailed log entry found | DESIGNED |
| R252c | K-FAC only 7B | File exists; no detailed log entry found | DESIGNED |
| R257 | Fixed batch FP / batch fixed rotation | File exists (two variants); no detailed log entry found | DESIGNED |
| R258 | Rank H_k 7B diagnostic (two variants) | File exists; no detailed log entry found | DESIGNED |
| R259 | Automaton vs WikiText comparison | File exists; no detailed log entry found | DESIGNED |
| R260 | CD ternary B,C | File exists; no detailed log entry found | DESIGNED |
| R265 | Cache incremental | File exists; no detailed log entry found | DESIGNED |
| R269 | Running XOR | File exists; no detailed log entry found | DESIGNED |
| R270 | Adaptive budget | File exists; no detailed log entry found | DESIGNED |
| R271 | Diagonal Newton | File exists; no detailed log entry found | DESIGNED |
| R271b | MiniBCD sweep | File exists; no detailed log entry found | DESIGNED |
| R272 | Integrated pipeline | File exists; no detailed log entry found | DESIGNED |
| R276 | TBBCD robust | File exists; no detailed log entry found | DESIGNED |
| R277 | OLS out_proj | File exists; no detailed log entry found | DESIGNED |
| R278 | W-shift | File exists; no detailed log entry found | DESIGNED |
| R279 | Parallel rounds | File exists; no detailed log entry found | DESIGNED |
| R280 | HSIC (Hilbert-Schmidt Independence Criterion) | File exists; no detailed log entry found | DESIGNED |
| R283 | Cache perturbation | File exists; no detailed log entry found | DESIGNED |
| R284 | Stale adjoint BCD | File exists; no detailed log entry found | DESIGNED |
| R287 | FP reverse | File exists; no detailed log entry found | DESIGNED |
| R288 | EGAC | File exists; no detailed log entry found | DESIGNED |
| R290 | OTSC | File exists; no detailed log entry found | DESIGNED |
| R291 | Semigroup test | File exists; no detailed log entry found | DESIGNED |
| R293 | Long seq | File exists; no detailed log entry found | DESIGNED |
| R294 | Gradient filter | File exists; no detailed log entry found | DESIGNED |
| R296 | Train and benchmark (most recent) | Loss 1.93 confirmed (from git log: "trained on real 2.96 MB dataset"), 166 tok/s | CONFIRMED |

## v12 Overnight Loop / Runtime Findings

- 2026-04-18 - `R386` full implementation closure: slices 1-4 were implemented and operationally validated, and Slice 5 closed the exact byte accounting plus canonical docs/state/queue updates. Exact steady-state omitted host bytes total: `487349760` (`~464.77 MiB`). Canonical evidence sources: `.work/jobs/r386-impl-01/validation.md`, `.work/jobs/r386-impl-02/validation.md`, `.work/jobs/r386-impl-03/validation.md`, `.work/jobs/r386-impl-04b/validation.md`.
- 2026-04-26 - `R364` implementation closeout: the `lm_head` pilot was implemented and stable on short batch-1 Vulkan probes, but Slice 2B showed no matched-objective speedup (`45.612s` baseline vs `74.318s` with `--r364-pilot`). The mathematician gate blocks expansion beyond `lm_head` under the current formulation, so the speed path is closed with the diagnostic code retained behind the disabled-by-default flag. Canonical evidence sources: `.work/jobs/r364-impl-01/validation-slice2b.md`, `.work/jobs/r364-impl-01/mathematician-gate.md`.

| # | Name | Result | Status |
|---|------|--------|--------|
| R364 | Upper-Subtree LS implementation pilot closeout | `lm_head` pilot implemented and stable on short batch-1 Vulkan probes, but Slice 2B showed no matched-objective speed signal; mathematician gate limits exact native expansion to `lm_head`; diagnostic code retained behind disabled-by-default flag | CLOSED_NO_GO |
| R368 M1 | Fused W1/sigmoid runtime cleanup + deterministic validation path | `S4-safe-gpu-only` landed and passed clean Vulkan rebuild + 10-step smoke; `train.c` gained `--seed`, `--log-every`, `--loss-log-hex`; two post-`S4` 100-step runs with `seed=42` matched on 100/100 `[S5]` lines; `N1` made the recompute path the default Vulkan backward path and preserved 100/100 `[S5]` equivalence versus the prior validated reference; `N2` then removed the remaining GPU forward-only gate cache path, kept zero hits for `fwd_a_eff_gate` / `a_eff_gate_out`, and preserved 100/100 `[S5]` equivalence across two new seed-42 runs; formal original `S5` remains blocked by the missing named pre-`S1` baseline | CONFIRMED |

| R399 SSQT | Sufficient-State Quotient Training gate | Synthetic fixture mechanics passed after correcting split-key coverage, but real opt-in Vulkan trace reached a non-identifying stop condition: build passed, default-disabled smoke emitted no R399 lines, env-gated trace activated and captured `768` rows with redundancy `4.571429`, but analyzer returned `INVALID` because raw gradient sketches were `768/768` zero with `0` nonzero records; no production integration and no gain claim; this blocks the current path but does not falsify SSQT theory | BLOCKED_BY_NON_IDENTIFYING_REAL_PATH |
| R399 DMDC | Deterministic MVUE Discrepancy Control gate | Real GPU/Vulkan trace build and smoke passed; default/no-env smoke emitted no `R399_DMDC` lines; opt-in trace activated in `mode=diagnostic-only` with `source=real-mvue-gpu-training`, `no_training_behavior_change=1`, and inequality `(1-f_q*(1-rho))*(1+h)<=0.50`; steps `0` and `1` captured `records=16384` total with `useful_records=0`; analyzer returned `decision=INVALID` with `reason=insufficient_useful_records`, `lhs/f_q/rho=NaN`, and zero useful energy denominators; synthetic fixture pass does not support production integration or speed claims; this blocks the current path but does not falsify DMDC theory | BLOCKED_BY_NON_IDENTIFYING_REAL_PATH |
| R400 TBET | Ternary Boundary Event Training / Discrete Flip Syndrome gate | Diagnostic-only Vulkan trace implemented at the real optimizer-to-MVUE boundary and one bounded active-EMA sampler repair was tested. Build passed; no-env smoke emitted no TBET lines; env-gated trace activated with `no_training_behavior_change=1`. Initial trace captured `8192` useful records but `nonzero_shadow_delta_count=0` with `actual_flip_count=1990` and `boundary_candidate_count=2018`, reason `all_zero_or_nan_shadow_deltas`. Repaired trace still returned `active_ema_record_count=0`, `nonzero_shadow_delta_count=0`, `actual_flip_count=1990`, `boundary_candidate_count=2018`, reason `no_active_ema_records`. Missing object: aligned real-path record with `abs(grad_accum_before) > ema_threshold`, `shadow_after != shadow_before`, and aligned `q`/`sampled` masks on the same boundary step. No production integration and no gain claim; TBET is parked, not theory-falsified | BLOCKED_BY_NON_IDENTIFYING_REAL_PATH |
| R400 CCHC | Credit-Cone Horizon Compression gate | Diagnostic-only Vulkan trace implemented at the real GPU SSM backward boundary and one bounded state-source repair was tested. Build passed; no-env smoke emitted no CCHC lines; env-gated trace activated with `no_training_behavior_change=1`. Initial trace had transition nonzero but zero state/adjoint/credit and analyzer reason `zero_state_l2`. Repaired trace used `state_source=layer.fwd_h_cache` with `state_source_valid=true`, captured `16` records, `transition_nonzero=2097152`, but `state_nonzero=0`, `adjoint_nonzero=0`, `credit_nonzero=0`, and analyzer returned `blocked_by_non_identifying_real_path` with reason `zero_state_l2`. Missing object: nonzero post-forward SSM state from `layer.fwd_h_cache` aligned with transition, adjoint, and `dA_ps` records. No production integration and no gain claim; CCHC is parked, not theory-falsified | BLOCKED_BY_NON_IDENTIFYING_REAL_PATH |
| R400 SOFSB | Self-Owned Future-State Bootstrapping gate | Diagnostic-only Vulkan trace implemented for current/future state and separated LM/bootstrap transition-gradient sketches. Build passed after moving the helper behind the full `GpuPipeline` definition; no-env smoke emitted no SOFSB lines; env-gated trace activated with `no_training_behavior_change=1`. Analyzer returned `blocked_by_non_identifying_real_path`, reason `comparable_records_below_minimum`: `records=16`, `comparable_records=0`, `state_nonzero_records=0`, `future_nonzero_records=0`, `lm_grad_nonzero_records=0`, `boot_grad_nonzero_records=0`, `median_cosine=0`, and zero variance proxies. Missing object: aligned nonzero record with `h_t`, `h_{t+k}` or future target, LM transition-gradient sketch, and bootstrap transition-gradient sketch for the same step/layer/microbatch/horizon/transition family `A`. No production integration and no gain claim; SOFSB is parked, not theory-falsified | BLOCKED_BY_NON_IDENTIFYING_REAL_PATH |
| R405 Q1 TCFVL | Ternary Counterfactual Flip Value Learning diagnostic gate | Standalone `r405_tcfvl_gate`, analyzer, Makefile target, queue repair, path sentinel `--path-check-only`, and target-aware candidate mode `--target-aware-candidates` were landed for the gate. Random smoke initially returned `INVALID_PROBE/all_zero_deltas`, but the path sentinel later proved the path identifying with `decision=PATH_IDENTIFYING` and `max_abs_delta=0.13507652282714844`. Final target-aware diagnostic activation was `[R405_TCFVL_GATE] active=1 mode=diagnostic-only source=v12-cpu-forward ce_mode=full contexts=64 batch_size=4 candidates=4 output_path=..\..\.work\jobs\r405-q1-tcfvl-gate-01\artifacts\r405_tcfvl_targetaware_trace.jsonl target_aware_candidates=1 no_training_integration=1`; analyzer returned `WEAK_SIGNAL` below thresholds with `record_count=8`, `valid_record_count=8`, `positive_count=3`, `negative_count=5`, `useful_delta_prevalence=0.375`, `precision_top10=0.375`, `precision_top10_ratio=1.0`, `auc=0.3`, `interaction_harm=-4.6591548765831435e-14`, `priced_ratio=0.5898070087339593`, `restore_failures=0`, `supported_families=[W_in,W_out,lm_head]`, `forward_eval_count=704`, and `baseline_ce=51.13909964985022`. This is a valid failed identifying gate, not an invalid probe, and supports no production integration or gain claim | FAILED_IDENTIFYING_GATE |
| R408 Corpus Scaffold | Corpus Scaffold + Residual CE native residual pilot and GPU opt-in landing | Standalone native C residual pilot measured original next-token CE with full residual count freedom. Easy matched target passed: baseline `0.277s` to CE `6.080136325208`, scaffold `0.227s`, `1.220265159273x`, CPU allocation estimate `537640448` bytes. Harder targets preserved residual-term savings but failed wall-clock: `12M -> 7M` terms at CE `5.849073213003` but `0.563s -> 0.623s`; `20M -> 16M` terms at CE `5.729831243270` but `1.004s -> 1.127s`. Production landing added an opt-in corpus bigram residual prior to the Model, CPU sampled-CE path, and Vulkan sampled-CE shader via `--r408-scaffold`, `--r408-scaffold-alpha`, `--r408-scaffold-scale`, `--r408-scaffold-terms`, and `--r408-scaffold-stride`. Vulkan uses sparse hash scaffold buffers and emitted `[R408_SCAFFOLD_GPU] active=1 mode=vulkan-sparse-hash-log-prior pair_capacity=65536 unique_pairs=1015 terms=1024` on a zero-step activation smoke. Build validation passed for `mingw32-make USE_VULKAN=1 shaders` and `mingw32-make USE_VULKAN=1 train`; activation smoke `train.exe test_data.bin 0 --batch 1 --seed 42 --log-every 1 --teaching-off --r408-scaffold --r408-scaffold-terms 1024` returned exit code `0` with `[VK] allocated: 728.73 MB`. GPU speed remains unmeasured and no GPU speedup is claimed yet. | GPU_PATH_LANDED_ACTIVATED |
| R409 Logit Prior Gate | CE-equivalent logit control variate real-path landing | Added mechanism-named production flags `--corpus-logit-prior` and `--logit-prior-gate`; Vulkan sampled-CE now records baseline sampled CE and residual-prior sampled CE in the same GPU pass. Build passed for `mingw32-make USE_VULKAN=1 shaders` and `mingw32-make USE_VULKAN=1 train`. Paired one-step RTX 5070 smoke on `test_data.bin`, batch `1`, seed `42`, teaching off: baseline `loss=11.7143`, `1231 tok/s`, `728 MB`; corpus logit prior + gate `loss=11.6785`, `1235 tok/s`, `729 MB`; gate line `[LOGIT_PRIOR_GATE] step=1 base_loss=10.379663 residual_loss=10.335491 residual_ratio=0.995744 batches=4`. Post-hygiene zero-step activation smoke with mechanism-named interface also passed and emitted `[CORPUS_LOGIT_PRIOR_GPU] active=1` on RTX 5070 with `728.74 MB` allocated before the zero-step loop. This validates activation and real-path measurement, but the tiny smoke is not enough to project a 10% training-time gain. | GPU_PATH_LANDED_ACTIVATED |
| R409 Program Kernel Gate | Proof-carrying deterministic transition kernel real-path landing | Added mechanism-named production flags `--program-kernel`, `--program-kernel-gate`, `--program-kernel-logit`, `--program-kernel-min-count`, `--program-kernel-terms`, and `--program-kernel-stride`; model build now creates an abstaining corpus-owned deterministic context-to-target table; CPU sampled CE and Vulkan sampled CE consume the kernel as an additive logit path; GPU gate records baseline sampled CE versus kernel sampled CE plus routed/correct/incorrect counts. Build passed for `mingw32-make USE_VULKAN=1 shaders` and `mingw32-make USE_VULKAN=1 train`. `test_data.bin` activation produced `entries=0 conflicts=256`, so it is a no-signal fixture for this one-token kernel. Synthetic exact-domain smoke on `.work/tmp/program_kernel_repeat_A.bin`, batch `1`, seed `42`, teaching off, `--program-kernel-logit 8.0`: `[PROGRAM_KERNEL_GPU] active=1`, `base_loss=46.534023`, `kernel_loss=38.534023`, `kernel_ratio=0.828083`, `routed=4092`, `correct=4092`, `incorrect=0`, `route_accuracy=1.000000`, VRAM `728.24 MB`. This validates the real GPU path and passes the material sampled-CE gate on an exact domain, but does not yet claim broad-corpus training-time gain. | GPU_PATH_LANDED_ACTIVATED |
| R409 State Transition Atlas Gate | Exact local state transition atlas real-path landing | Added mechanism-named production flags `--state-transition-atlas`, `--state-transition-atlas-gate`, `--state-transition-atlas-logit`, `--state-transition-atlas-min-count`, `--state-transition-atlas-terms`, and `--state-transition-atlas-stride`; model build now creates an abstaining corpus-owned `(previous_token,current_token)->target` transition atlas; CPU sampled CE and Vulkan sampled CE consume the atlas as an additive logit path; GPU gate records baseline sampled CE versus atlas sampled CE plus routed/correct/incorrect counts. Build passed for `mingw32-make USE_VULKAN=1 shaders` and `mingw32-make USE_VULKAN=1 train`. Paired one-step RTX 5070 smoke on `test_data.bin`, batch `1`, seed `42`, teaching off: baseline `loss=11.7143`, `1253 tok/s`, `727.99 MB`; atlas gate `loss=9.4774`, `1258 tok/s`, `728.74 MB`; gate line `base_loss=10.379663`, `atlas_loss=2.901261`, `atlas_ratio=0.279514`, `entries=3857`, `routed=1321`, `correct=1158`, `incorrect=163`, `route_accuracy=0.876609`. This validates real GPU activation and a strong low-scale sampled-CE signal, but does not yet claim broad-corpus training-time gain. | GPU_PATH_LANDED_ACTIVATED |
| R409 Counterfactual Packets Gate | Counterfactual minimal-pair packet real-path landing | Added mechanism-named production flags `--counterfactual-packets`, `--counterfactual-packets-gate`, `--counterfactual-packets-min-count`, `--counterfactual-packets-terms`, and `--counterfactual-packets-stride`; model build now creates a corpus-owned local-state packet ledger storing the first two observed next-token targets for `(previous_token,current_token)` states; the GPU gate records packet coverage and balance while leaving original CE unchanged. Build passed for `mingw32-make USE_VULKAN=1 train`. One-step RTX 5070 smoke on `test_data.bin`, batch `1`, seed `42`, teaching off: activation emitted `pair_states=118`, `[TRAIN] step=0 loss=11.7143`, `1258 tok/s`, `727.99 MB`, and gate line `packets=65`, `primary=5`, `secondary=60`, `other=3`, `abstain=4020`, `coverage=0.052838`, `balance=0.074074`. This validates real GPU activation and nonzero packet signal, but coverage/balance are weak and no broad-corpus training-time gain is claimed. | GPU_PATH_LANDED_ACTIVATED |
| R409 Certified Silent-Span Gate | CE-gradient-contained silent-span real-path landing | Added mechanism-named production flags `--certified-silent-span-gate`, `--certified-silent-span-max-loss`, `--certified-silent-span-min-len`, `--certified-silent-span-cert-cost`, and `--certified-silent-span-repair-cost`; the Vulkan CE path now records a conservative silent-span certificate from per-token sampled CE and reports `silent_fraction`, `grad_bound`, and `priced_ratio` while leaving original CE unchanged. Build passed for `mingw32-make USE_VULKAN=1 train`. One-step RTX 5070 smoke on `test_data.bin`, batch `1`, seed `42`, teaching off: activation emitted `[CERTIFIED_SILENT_SPAN_GATE] active=1`, `[TRAIN] step=0 loss=11.7143`, `1258 tok/s`, `727.99 MB`, and gate line `tokens=0`, `positions=4092`, `spans=0`, `fraction=0.000000`, `grad_bound=0.000000`, `priced_ratio=1.000000`. This validates the real GPU certificate hook, but the conservative early fixture exposes no silent mass and no training-time gain is claimed. | GPU_PATH_LANDED_ACTIVATED |
| R409 Error-Causal Lattice Gate | Coverage-conserving prerequisite lattice real-path landing | Added mechanism-named production flags `--error-causal-lattice`, `--error-causal-lattice-gate`, `--error-causal-lattice-min-count`, `--error-causal-lattice-terms`, and `--error-causal-lattice-stride`; model build now creates target/prerequisite atom support counts and the GPU gate records target coverage, prerequisite readiness, orderable fraction, and rare-target coverage while leaving sampling and original CE unchanged. Build passed for `mingw32-make USE_VULKAN=1 train`. One-step RTX 5070 smoke on `test_data.bin`, batch `1`, seed `42`, teaching off: activation emitted `target_atoms=256`, `prereq_atoms=256`, `[TRAIN] step=0 loss=11.7143`, `1266 tok/s`, `727.99 MB`, and gate line `positions=4092`, `target_covered=4092`, `prereq_ready=4092`, `orderable=4092`, `target_coverage=1.000000`, `orderable_fraction=1.000000`. This validates the real GPU prerequisite-lattice hook, but no scheduling-speed gain is claimed. | GPU_PATH_LANDED_ACTIVATED |
| R419 TSSI | Ternary Sufficient-Statistic Import CPU pre-gate, residual pilot, ssm_state proxy, and live Vulkan SSM trace gate | CPU-only `scripts/research/r419_tssi_gate.py` compiled corpus CE moments from real v12 byte-level shards into ternary sign/zero readout commitments. Main `wiki_1mb.bin` pre-gate used `500000` tokens with `300000/100000/100000` train/calib/held-out split and selected bigram features, density `0.05`, scale `2.4`: held-out CE `2.949221092857901` vs uniform `5.545177444479562` and random held-out CE mean `5.722178622339609`; weighted sign alignment `0.9986051321029663` vs random `0.45805180817842484`. Residual original-CE pilot in the same byte/bigram readout-like scope used seed `777`, target speedup vs best control `3.5455508909238374` and same-budget CE advantage `0.2352878570289607`; sign-shuffled control did not reach target. Deterministic recurrent `ssm_state` proxy on `500000` tokens also passed: seed `777` target speedup `3.6053478120822366`, same-budget CE advantage `0.05614244344347741`; seed `419` target speedup `3.5931313221153545`, same-budget CE advantage `0.05664628606735311`. Live Vulkan SSM trace gate repaired R395 trace materialization from `vulkan_fwd_h_cache`; a `10`-step batch-1 trace had `7864320 / 7864320` nonzero state values. Best `all`-layer TSSI import: held-out CE `5.346169927602639` vs random held-out CE mean `5.54442637453797`, weighted sign alignment `0.9970957040786743`, residual same-budget CE advantage `0.19843119959677402`, but no matched-target speedup because charged trace/export/build time dominates and controls do not reach the imported target. Positive partial signal only: no integrated production GPU import path, baseline change, or paper-ready claim yet. | PARTIAL_SIGNAL_LIVE_TRACE_CE_ONLY |

R419 closeout addendum (2026-05-11): `scripts/research/r419_live_trace_gate.py`
now supports `--split-mode records` and `--record-rotation`. Record-disjoint
live-trace rotations passed all five held-out record pairs: all-layer CE margin
vs best density-matched random min had min `0.189190727524743`, mean
`0.204293772528715`, and max `0.238266453598485`; `last` and `mean` layer
summaries also passed all rotations with weaker mean margins
`0.0327855562301441` and `0.013581927197886`. Updated status:
`PAPER_CANDIDATE_LIVE_STATE_POSITIVE_NOT_SPEED_CLAIM`; still no production GPU
speedup, baseline change, or paper-ready speedup claim.

| R422 Original-CE Credit Debt | CPU gate plus real v12 Vulkan observer for local direct original-CE transition credit. CPU gate passed the local-direct submechanism on five real-token smokes with projected whole-step estimates `1.229014x` to `1.243235x`, but sparse active-surface certification failed. The opt-in real v12 observer activated with `objective=unchanged_ce`, `observer_only=1`, and `no_training_change=1`; current live `v_pre * h_prev` embedding failed on `test_data.bin`, `text_data.bin`, and `wiki_1mb.bin` with local progress about `0.192524` to `0.210093` and priced ratios `2.534235` to `2.764009`. Status: `partial_signal_cpu_local_direct_positive_realpath_embedding_negative`; no speedup, no baseline change, no production training behavior. | PARTIAL_SIGNAL_REALPATH_CURRENT_EMBEDDING_NEGATIVE |
| R422 Reversible State Ledger | CPU structured binding gate for proof-carrying reversible memory. Main gate passed with decode fraction `0.999685`, CE gain vs keyed same-state `1.190161`, CE gain vs bitmatched larger state `0.602692`, CE gain vs 4x larger state `0.338157`, and zero false/corrupt false accepts. High-load control failed with decode `0.952444`; short-key dictionary control failed because a full dictionary became cheaper (`dictionary_memory_ratio=0.562855`). Status: `positive_structured_binding_gate_scope_limited`; no production v12 path, no speedup, no broad quality baseline change. | POSITIVE_STRUCTURED_GATE_SCOPE_LIMITED |
| R422 Lossless Packet CE | CPU real-shard packet gate for reversible macro original-CE units. Exact reconstruction and zero chain-rule identity error held in all retained runs. `text_data.bin` passed with packet ratio `0.126667`, priced ratio `0.186660`, macro coverage `1.000000`, shuffled margin `0.439863`; `cyberjutsu` passed with packet ratio `0.304200`, priced ratio `0.381289`, margin `0.196056`. `wiki_1mb`, `books_tokenized`, and `combined_10` failed strict shuffled/count controls; `test_data` had no macro mass. Status: `partial_signal_structured_text_packet_gate_not_general`; no production path, speedup, or broad baseline change. | PARTIAL_SIGNAL_STRUCTURED_PACKET_GATE |
| R422 Capability Closure | CPU verifier gate for normal-rule closure plus minimal causal intervention ledgers. The mid-budget `480` run passed with score gain `0.340043`, total gain `0.098705`, exception gain `0.788242`, normal regression `0.004854`, and capability-per-budget ratio `1.529262` against the best control `closure_without_exception_ledger`; four additional budget-480 seeds also passed. Budget `240` failed by harming normal accuracy, and larger/looser settings failed strict total-gain thresholds as controls caught up. Status: `positive_mid_budget_capability_closure_gate_scope_limited`; no production path, speedup, or broad baseline change. | POSITIVE_SCOPE_LIMITED_CAPABILITY_GATE |
| R422 Executable Micro-Theory | CPU verifier gate for priced executable local-theory objects versus ordinary examples and trace supervision. After correcting inference-cost pricing for controls, main/reuse/seed runs passed against best control `trace_supervision` with total gain `0.138802` to `0.187760`, long-composition gain `0.160821` to `0.236332`, correct-per-paid ratio `1.946677` to `2.357733`, and exact state checks `1.000000`; budget `320` failed because object coverage was insufficient. Status: `positive_executable_micro_theory_gate_amortized_scope_limited`; no production path, speedup, or broad baseline change. | POSITIVE_AMORTIZED_MICRO_THEORY_GATE |
| R422 Event-Causal Credit Ledger | GPU/v12 closeout using existing event-adjacent gates plus a bounded chamber/flip activation smoke. The event smoke activated `--ce-normal-chamber-gate`, `--r388-flip-refresh-gate`, and `--optimizer-integrity-gate`; optimizer integrity passed, but the early chamber had `saturated=0`, `candidates=0`, `priced_k16=0`, and `identifying=0` at `740MB` VRAM. Prior R416/R417/R388 evidence shows boundary-anchor credit fails, direct TSA/gbar evidence commitments are harmful, variance-priced credit is aligned but score/verify pricing-blocked, and flip-refresh lacks valid mature evidence. Status: `parked_blocked_by_non_identifying_or_pricing_blocked_event_path`; no speedup, no baseline change, no production training behavior. | PARKED_EVENT_CREDIT_GPU_PATH_NOT_PROMOTED |

