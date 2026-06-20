# Project Manifesto

This project is not a local-model optimization exercise. It is a research
program for changing the training economics of modern language models.

## Objective

Build a project-owned model that can be trained from scratch on consumer
hardware and reach the practical quality class of strong small modern models
such as Qwen/Gemma-class systems, while remaining fast and light enough to run
locally.

The target is not to copy Qwen. Qwen is a quality reference point. The project
must discover a different path that preserves or improves quality while using
far less hardware.

## Why The Research Exists

The project already made major progress:

- training large SSM-style models can fit in consumer VRAM through
  layer-sequential / FP-style execution and compressed frozen state;
- inference and storage are structurally lighter than a comparable dense
  transformer target;
- the system has moved beyond PyTorch prototypes into C / native production
  paths;
- several papers and original formulas already exist in the project history.

The unsolved frontier is not "can it fit". The frontier is:

1. reduce time-to-quality from months toward days or hours;
2. preserve original CE / final task quality;
3. keep inference local, fast, and memory-light;
4. avoid moving cost into hidden teachers, proxy objectives, repair systems, or
   unpriced preprocessing.

## User Mental Model

The user is looking for the airplane, not a faster car.

Next-token prediction is a current substrate, not a sacred objective. The real
goal is an intelligent local system that does what is needed as quickly and
cheaply as possible. If a different mathematically grounded substrate reaches
the same or better capability with less training work, it is in scope.

Therefore:

- do not anchor only on existing LLM recipes;
- do not bring mainstream web-search recipes and bolt them onto the model;
- do not turn the system into a Frankenstein stack of unrelated techniques;
- start from mathematics and full-path objective equivalence;
- use paradigm reframing to escape local minima in the current design;
- then derive algorithms, then code, then gates.

## Research Standard

A positive research result must be more than an elegant idea.

It must answer:

- what mechanism changes the training/inference economics;
- what mathematical object or invariant preserves quality;
- what exact cost denominator is reduced;
- what hidden costs are fully priced;
- what could kill it quickly;
- why that quick kill does not already apply.

Validated means the conceptual failure modes are removed. Remaining risk may be
ordinary implementation bugs, not "maybe the idea is false".

## Non-Negotiables

- No quality tradeoff disguised as speed.
- No external stronger conventional teacher as the main solution.
- No proxy objective unless it is exactly equivalent to original objective or
  has a proven harmless residual.
- No unpriced preprocessing, routing, scoring, repair, search, or verification.
- No broad implementation before a cheap proof/gate says the finding can live.
- No round closure with zero findings unless the no-go is itself a strong scoped
  research result or the user explicitly stops.

## Preferred Thinking Order

1. Define the real target and artifact boundary.
2. Write the mathematical objective and full priced path.
3. Look for an exact quotient, sufficient statistic, coordinate change,
   conserved quantity, certificate, or description that removes work.
4. Ask whether the whole substrate is wrong.
5. Derive the smallest algorithm that implements the mathematical object.
6. Run the cheapest identifying gate.
7. Only then implement broadly.

## How To Talk About Progress

Separate every claim into:

- measured with an RXXX source;
- projected from an RXXX source;
- validated-for-gate;
- candidate;
- blocked / non-identifying;
- rejected.

Never flatten those categories into one confidence level.

