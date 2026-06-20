# Strategic Direction

**Last updated:** 2026-05-21

## Objective

The project objective is not paper production by itself. Papers are expected
outputs when the project discovers something real, but the primary objective is:

- train a project-owned modern-quality model on consumer hardware;
- make training fast enough to be practical on a single strong consumer GPU;
- reach quality comparable to or better than contemporary strong open models;
- preserve local inference as a first-class product constraint, including high
  tokens-per-second when the trained model is run locally.

## User Objective Refresh - 2026-05-05

The concrete target is a project-owned model in the practical class of recent
strong small/open models such as Qwen3.6 and Gemma 4: small enough to run
locally, strong enough to feel modern, and trainable by the user on consumer
hardware. The goal is not a quality tradeoff for efficiency. The target is a new
LLM training and inference paradigm where high quality, very high local
inference throughput, low RAM/VRAM use, and fast training on a single consumer
GPU are mutually reinforcing consequences of the formulation.

Research should therefore treat "local first" as a full-stack invariant:

- training must be fast enough for consumer-hardware iteration, not only for
  datacenter pretraining;
- inference must remain extremely fast and memory-light after training;
- quality cannot be purchased by outsourcing the hard work to a larger external
  conventional teacher;
- any mechanism that works locally should plausibly scale upward to datacenter
  settings as a paradigm improvement rather than a local-only workaround.

The research direction should therefore prefer discoveries that change the
training paradigm, model geometry, teaching signal, credit assignment, or
capability acquisition path. Code-only optimization is useful only when it
unlocks or validates such a discovery. It is not the default research frontier.

## Active Research Lines - 2026-05-21

The project now has three explicit research lines. Future work must route to the
right line before research, implementation, benchmarks, or documentation.

### Transformer Consumer Line

This line is active when the user asks for conventional models, Transformers,
open models, or consumer-machine improvements.

- Scope: existing Transformer-family models and conventional open-model
  ecosystems.
- Exclusions by default: no Mamba, no SSM, no project-native ternary SSM, and
  no v12 architecture work unless the user explicitly asks for cross-line
  comparison or reuses a project-native idea inside a Transformer-compatible
  formulation.
- Objective: make existing Transformer models train, fine-tune, serve, or run
  better on consumer hardware without hiding cost in unrealistic infrastructure.
- Primary denominators: VRAM, RAM, wall-clock training or fine-tuning time,
  tokens per second, context cost, optimizer state, activation memory, KV-cache
  memory, data efficiency, and quality per paid compute.
- Acceptable surfaces: attention, KV cache, quantization, activation checkpoint
  strategy, optimizer/state compression, low-rank or sparse adaptation,
  tokenizer/data strategy, context handling, batching, inference engines,
  fine-tuning recipes, and local deployment constraints, as long as the work is
  tied to real consumer hardware.
- Evidence standard: current external facts about models, libraries, kernels,
  or tools must be verified from current sources before being used as claims.

Routing signals for this line include `Transformer`, `modelos convencionais`,
`open model`, `Llama`, `Qwen`, `Gemma`, `Mistral`, `DeepSeek`, `fine-tuning`,
`LoRA`, `quantization`, `inference`, `serving`, `consumer GPU`, and similar
language.

### Project-Native Ternary SSM Line

This line is paused by default after the 2026-05-20 pivot. It is not abandoned.
It remains the long-horizon project-native architecture line for training a new
model formulation from scratch on consumer hardware.

Resume this line only when the user explicitly asks for `ternary SSM`,
`Mamba/SSM`, `v12`, `R4xx`, `project-owned architecture`, `our model from
scratch`, or another clearly project-native context.

### Constructive Imagination And Growth Line

This line is active when the user asks for imagination, internal world/object
construction, database-like representational substrates, spatial coherence,
physical/mechanical consistency, open-ended teachability, non-fixed parameter
growth, or child-like interactive learning.

- Scope: a new model substrate whose core capability is constructing,
  inspecting, repairing, and growing internal object states before answering.
- Exclusions by default: not ordinary RAG, not a bigger Transformer recipe, not
  just a multimodal generator, not a prettier 3D asset generator, and not
  automatically the ternary SSM line.
- Objective: create an imagination layer that preserves object identity, rough
  geometry, constraints, affordances, and teachable growth over time.
- Primary denominators: physical/spatial violation rate, identity preservation
  across edits, repair sample efficiency, sparse update cost, teach-test
  retention, memory growth, and consumer-hardware update cost.
- Initial seed: `.workspace/research/2026-05-21-imagination-growth-line-seed.md`.

### Ambiguous Context Rule

If the user request could plausibly mean more than one line, ask which line
before research or implementation. Do not silently continue the ternary SSM queue
when the user is asking for conventional Transformer work, and do not collapse
the imagination line into either existing line unless the user explicitly asks
for cross-line comparison.

## Current Finding Classes

### Primary Discovery Queue

These findings are the current best fit for the project's real direction:

| finding | aspect | projected gain | why it stays primary |
|---|---:|---:|---|
| `R427` Unified Growth-Ladder System | no-reset growth / residual credit / local artifact | no measured projection change yet | Best current fit to the real target: train a small lineage upward instead of restarting at 7B, while preserving old behavior, residual child learning, and local inference artifact shape. |
| `R427` Segment-Credit Certificates | original-CE work / recurrent credit | no measured projection change yet | Strongest new route to reduce recurrent backward work without repeating the failed endpoint-only boundary mistake. |
| `R427` Capability-Dense Training Material | quality per token / capability acquisition | no measured projection change yet | Attacks the Qwen-class quality target by making each paid token carry more project-owned capability signal, without relying on a stronger external teacher. |
| `R394 QCF` Quantization-Cell Quotient Flow | speed / ternary geometry | `~1.6-2.3x` wall-clock-to-target | Directly targets the quotient geometry of ternary quantization cells and can become a project-specific training formulation. |
| `R394 SSH` Spectral-Stiffness Homotopy | speed / training dynamics | `~1.8-2.7x` wall-clock-to-target | Changes the spectral path of the SSM during training rather than optimizing runtime mechanics. |
| `R394 BCT` Boundary-Condition Credit Transport | speed / credit assignment | `~1.7-2.6x` wall-clock-to-target | Reframes internal state targets and downstream credit, which is aligned with model-owning training research. |
| `R394 RBTC` Rao-Blackwellized Ternary Credit | speed / estimator variance | `~1.5-2.1x` wall-clock-to-target | Uses conditioning and variance reduction specific to ternary credit rather than a generic software optimization. |
| `R390` Evidence-Demand and Deferral Episodes | datasets / teaching | `~1.5-2.0x` time-to-capability | Teaches evidence-seeking and deferral control, which improves the behavior of a project-owned model. |
| `R390` Intervention-Outcome Teaching Episodes | datasets / teaching | `~1.6-2.3x` time-to-capability | Adds action-consequence supervision and grounded correction. |
| `R390` Capability-Deficit Graph Tutoring | datasets / teaching | `~1.5-2.4x` time-to-capability | Reframes data selection around live capability deficits and prerequisites. |

Prior teaching survivors `F1`, `F2`, `F3+`, and `F4` remain active but need
their original handoffs opened before they are placed ahead of the `R390` and
`R394` lines.

### Product / Runtime Parking Lot

These lines may still help a shipped local system, but they are not the default
research frontier because they are mostly runtime or implementation mechanics:

| finding | status | reason |
|---|---|---|
| `R370` DAPSK persistent-threads mega-kernel | parked runtime/product | Kernel dispatch engineering. Useful if product throughput becomes the bottleneck, but not a model-owning discovery by itself. |
| `R388` Flip-Event Sparse Refresh | parked runtime/product | Scheduler/cache-invalidation style speed line. More algorithmic than `R370`, but still closer to runtime product work than core training theory. |

### Closed / No-Go

| finding | status | reason |
|---|---|---|
| `R364` Upper-Subtree LS | closed no-go under current formulation | The `lm_head` pilot was stable but slower; exact native expansion is limited to `lm_head`; no further speed work without a new CE-aligned tangent formulation. |

## Anti-Pattern: Code-Heavy Implementation Before Gain Is Proved

The R364 and R370 attempts exposed the same workflow risk from different sides:

- R364 reached the mathematical boundary too late, after expensive debugging.
- R370 began as a runtime implementation path even though its best outcome was
  mostly software dispatch optimization, not a project-level discovery.

For validated findings, implementation must now start with the cheapest
gain-validating gate:

1. Restate the theoretical mechanism and what would make it original.
2. Derive the minimum inequality or invariant that must hold for the projected
   gain to be possible.
3. Identify the smallest measurement, proof, or synthetic probe that can kill
   the finding before broad implementation.
4. Run that gate before any large code path, mega-kernel, scheduler rewrite, or
   multi-file production integration.
5. Validate the gate itself before treating the result as a finding decision.
6. If the gate is identifying and fails, close or park the finding immediately
   and document why.

Broad implementation is justified only after the gain mechanism survives this
gate. Passing a build or smoke test is not enough; the claimed gain must remain
plausible under the measured or derived invariant.

### Anti-Pattern: Invalid Probe As Finding Kill

The first R390 IOTE probe exposed a second workflow risk: a falsifier can fail
because the probe cannot express the mechanism, not because the mechanism is
false.

For every kill-fast probe, the closeout must explicitly answer:

- could the candidate arm have produced a positive primary metric if the theory
  were true?
- did any arm produce nonzero signal on the primary metric?
- were candidate, baseline, and ablation arms given only their allowed
  information?
- does an ablation tie mean real collapse, or did all arms fail vacuously?
- did the fixture require an extra capability not supplied by the probe, such as
  copying or rebinding held-out arguments?

If those checks fail, the correct result is `invalid_probe` or
`parked_pending_corrected_validation`. Do not mark the finding `killed` until a
corrected, identifying probe fails.

## Next Research Move

The next research move should be a focused theoretical comparison of the `R394`
family, with `R394 QCF` as the first candidate to inspect deeply because it is
the most project-specific to ternary training geometry.

The comparison should rank candidates by:

- originality of the formulation;
- probability of improving wall-clock-to-target rather than only runtime
  throughput;
- direct relevance to a project-owned model on consumer hardware;
- ability to produce a minimal falsification gate before implementation;
- compatibility with local high-throughput inference after training.
