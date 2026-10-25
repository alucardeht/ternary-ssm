# R508–R510 Continuation Summary

**Dates:** 2026-09-26 to 2026-10-25
**Status:** R510 closed. The BSR model is validated within its curated grammar and ready for optimization.
**Where the work lives:** a separate research workspace (rounds R508–R510). The code of this continuation is not in this repository yet; this file records what was measured and how far each claim goes.

Every claim carries one evidence class:

| class | meaning |
|---|---|
| MEASURED | observed in a real run with result files |
| GATE_PASS | survived a cheap falsifier (algebra, construction or a fixture), not end to end |
| CANDIDATE | proposed, not yet validated |
| REJECTED | the identifying test failed |

---

## Why the direction changed

The June release left two barriers open for the ternary SSM: training time (60B tokens ≈ 495 days on an RTX 5070) and the Forward Programming quality gap. The continuation first re-tested ideas on the released material (R508, R509). R510 then asked a different question: can a model learn far faster with far less compute if knowledge does not have to live in the weights? The answer it reached is a different architecture, not a faster ternary SSM. **The 495-day barrier of the ternary SSM is untouched.**

All continuation work ran on CPU only (Apple M4 Pro laptop, 2 threads, at most about 1 GB of RAM and 1 hour of training per test), with tiny models.

---

## R508 — idea tests on the released material (2026-09-26/27)

| claim | class | number |
|---|---|---|
| composition LOCK-RC2: ternary weights with an analytic STE, SSM with A ∈ {±1}, SWISS keep-set (window ∪ oracle), MoE with keep routing | GATE_PASS | cost identities and fixtures |
| neural induction with a non-local keep-set | GATE_PASS | 0.83 vs sliding window 0.05 vs random keep 0.43 |
| structured world state beats captions (same-caption twins) | GATE_PASS | 1.0 vs 0.05 |
| Choice / Jev router | REJECTED | in that setup |

Not shown: a language-modelling (cross-entropy) gain on a real model; kernel latency.

## R509 — agent action form: a regional ternary SSM (2026-09-27 to 2026-10-22)

| claim | class | number |
|---|---|---|
| regional ternary SSM v1.3 end to end | MEASURED (seed 0) | recall 0.99, edit 0.97, hop2 0.87, cross-session 0.98, conformal selective speech |
| consolidation as a delta memory in list form | MEASURED | cross-session 0.951 with the in-session memory cleared, no retraining |

Limits: one seed; curated templated data.

## R510 — memory capacity, a brain study, and the BSR model (2026-10-23 to 2026-10-25)

### Phase A: the R509 store

| claim | class | number |
|---|---|---|
| top-k superposition over snapshot records | MEASURED | recall 0.932 / 0.896 / 0.958 at 1 / 20 / 50 sessions (the θ-pruned list decayed to 0.841 at 20) |
| type-gated cross-session replacement | MEASURED | edit 0.975, untouched same-entity fact 0.875 |
| increment records instead of snapshots | MEASURED | hop2 0.49 → 0.79 at one session |

A matched pure-text transformer baseline was planned and dropped before evaluation.

### Phase B: brain study

28 topic notes over 126 sources (energy budget, coding, hippocampus, consolidation and sleep, basal ganglia, cerebellum, prefrontal working memory, neuromodulators, language, vision), a synthesis of ten cross-cutting motifs, derivations and one hypothesis: knowledge in a one-shot event memory, a small reader in front of it, offline consolidation behind it.

| claim | class | number |
|---|---|---|
| block-sparse binding in Z_64^32 | GATE_PASS | exact unbinding, Bloom-law bundling (algebra known in the VSA literature) |
| one-shot binary association layer | GATE_PASS | capacity law exact; with depth and replay, old replayed memories kept 0.78 (0.02 without) while new ones are learned 0.99 |

### Phase C: the BSR model (block-sparse regional)

**What it is.**
- An event memory that stores each fact in one write (role-bound block codes), completes patterns, edits by recency and abstains when it does not know.
- A reader that learns role tags and clause boundaries by one pass of counting (Witten–Bell back-off, beam decoding with one frame per clause). No gradient.
- A speaker that realizes each proposition from counted templates.
- Offline consolidation into rules and concepts nobody stated, with exceptions kept; a habit layer; a pointer working memory for pronouns; neuromodulator-like scalars for change detection; an acting agent on a grid world; a synthetic retina writing into the same memory.
- A curator that, after each conversation, decides by itself what becomes permanent (keep, hold, ask, drop), with calibrated probabilities and trust learned per source.
- New sentence forms learned overnight from the model's own memory: sentences of the day about facts it already knows label the new form, and the night counts it, re-reads the day and only then curates.

The whole model runs without any trained network. A ternary SSM reader (84.5k parameters) remains available as an option and is the comparator below.

**Results** (MEASURED; comparator: the project's ternary SSM reader r6 on the same curated Portuguese grammar; anti-memorisation controls on every test: pseudo-word lexicons, held-out syntax, a memory-cleared control):

| test | BSR, counting reader | ternary SSM reader |
|---|---|---|
| recombination of held-out patterns, n = 9 / 18 / 36 | 0.639 / 0.843 / 0.982 | 0.317 / 0.543 / 0.754 |
| in distribution, unseen lexicon | 1.000 | 0.965 |
| 3 / 4 / 5 clauses, unseen lexicon (fresh sets) | 0.950 / 0.935 / 0.930 | 0.875 / 0.765 / 0.725 |
| never-seen fillers | 0.86 | 0.785 |
| negation | 1.00 | 0.965 |
| end to end on pseudo-words (facts, edits, chains, definitions, pronouns, consolidation) | 1.00 | 0.65–0.98 |
| reader training | 0.67 s for 10,000 sentences, one pass | 26 min for 80,000 sentences |
| operations at matched reading accuracy (about 0.95) | about 2·10⁵ counter writes | about 1.6·10¹¹ dense-equivalent operations |
| speaker training | 50 examples, 3 ms | 48,000 examples, 7 min |

| capability | class | number |
|---|---|---|
| new sentence form learned overnight from memory | MEASURED | facts in the new form answered 1.00 after the night and the next day, 0 wrong facts, 0 junk (form learning off: 0.08) |
| curated learning while conversing | MEASURED | precision 0.975 vs 0.899 storing everything, at equal recall 0.919; lies from an intruder stored 2 of 60 vs 40%; calibration error 0.044 |
| memory scale | MEASURED | answers flat at 0.15 ms from 1k to 300k events |
| consolidation | MEASURED | rules and concepts never stated, 0 false inference at ≥ 60 entities |
| lifetime-valid night (anytime-valid consolidation) | GATE_PASS | removes the 10–12 false concepts per life the nightly test adopts, 2–4 nights later; not integrated |
| natural multi-fact sentences as memory-labelled fact graphs | CANDIDATE | experimental, off by default |

**Where it is not better.**
- Long sentences with the known lexicon: 0.93–0.96 against 1.00 for the ternary reader.
- Off-grammar zero-shot reading of hand-written natural sentences: 0.22 / 0.145 against 0.28 / 0.177.
- Zero-shot word orders that move a role: 0 for both; the route measured so far is learning the form from memory overnight.

**Validation.** Nine independent review passes. The eighth found the model validated within scope (no claim contradicted, every limit documented) and ready for optimization; after its wording fixes, the pinned set was rerun and compared cell by cell: 17,102 cells, 0 differences. The ninth reproduced 21,457 cells and left the default model unchanged.

**Novelty** (GATE_PASS: a prior-art audit, not a systematic review). No published system was found that combines gradient-free compositional reading, sentence forms learned from unannotated speech through the system's own fact memory with an explicit held state, a night that re-reads the day before a trust-learning curator, and a conversational learner. Every ingredient is prior art; the closest systems are NELL, Thomason et al. (2015, 2020), Fluid Construction Grammar tutor-learner agents and DreamCoder. The identifiability conditions behind the form learning are restatements of known results.

**Rejected along the way** (REJECTED, with the measurement): the lexical role memory as a decoding prior (in-session recall 0.98 → 0.89); skipping unknown words in the reader's context (recombination 0.976 → 0.824); learning forms by refutation (unsound on the model's relations); a completeness law proposed for form learning (an identity, already known).

---

## Not claimed

- No comparison with transformers, LLMs or published baselines on public benchmarks. The only trained comparator is this project's own ternary SSM reader.
- No open-language competence: every language result is on a curated templated grammar whose structure the reader exploits.
- Nothing about training a large model, and nothing about the 495-day barrier.

## Open

1. Optimization of the validated model (the counting reader's Python beam, about 1 ms per token, is the hot spot).
2. Reading natural, non-templated language.
3. An independent review of the fact-graph forms on a fresh slice.
4. Integrating the lifetime-valid night and a render-and-verify check for exact visual detail.
5. Vision beyond the synthetic retina.
