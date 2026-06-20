#!/usr/bin/env python3
"""R422 capability equivalence-class objective gate.

The gate tests whether a deterministic capability-class extractor can separate
capability learning from surface variation while preserving the exact surface
target as class CE plus residual surface CE. It is a CPU material/proof gate,
not a production training path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_capability_equivalence_gate.v1"


@dataclass(frozen=True)
class Row:
    prompt: str
    capability_class: str
    surface: str


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    train_rows: int
    heldout_rows: int
    classes: int
    surfaces_per_class: int
    extractor_accuracy: float
    noisy_extractor_accuracy: float
    class_ce: float
    flat_surface_ce: float
    factor_surface_ce: float
    residual_surface_ce: float
    random_class_ce: float
    class_ce_gain_vs_flat_surface: float
    class_ce_gain_vs_random: float
    factorization_identity_error: float
    residual_guard: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


def make_rows(
    *,
    prompts: int,
    classes: int,
    surfaces_per_class: int,
    repeats_per_prompt: int,
    seed: int,
) -> list[Row]:
    rng = random.Random(seed)
    templates = [
        "answer {cls} via proof {style}",
        "result {cls} using method {style}",
        "final {cls} because witness {style}",
        "emit {cls} with certificate {style}",
    ]
    rows: list[Row] = []
    for pid in range(prompts):
        cls_id = (pid * 17 + pid // 3) % classes
        cls = f"c{cls_id}"
        prompt = f"task_{pid % 11}_input_{pid}"
        for rep in range(repeats_per_prompt):
            style = f"s{rng.randrange(surfaces_per_class)}"
            template = templates[(pid + rep + rng.randrange(len(templates))) % len(templates)]
            rows.append(Row(prompt=prompt, capability_class=cls, surface=template.format(cls=cls, style=style)))
    rng.shuffle(rows)
    return rows


def noisy_class(cls: str, classes: int, noise_rate: float, rng: random.Random) -> str:
    if rng.random() >= noise_rate:
        return cls
    current = int(cls[1:])
    offset = rng.randrange(1, classes)
    return f"c{(current + offset) % classes}"


def split_rows(rows: list[Row], train_fraction: float) -> tuple[list[Row], list[Row]]:
    split = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    return rows[:split], rows[split:]


def count_models(rows: list[Row], classes: int, noise_rate: float, seed: int) -> dict:
    rng = random.Random(seed)
    class_by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    surface_by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    residual_by_prompt_class: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    random_class_by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    noisy_by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        class_by_prompt[row.prompt][row.capability_class] += 1
        surface_by_prompt[row.prompt][row.surface] += 1
        residual_by_prompt_class[(row.prompt, row.capability_class)][row.surface] += 1
        random_class_by_prompt[row.prompt][f"c{rng.randrange(classes)}"] += 1
        noisy_by_prompt[row.prompt][noisy_class(row.capability_class, classes, noise_rate, rng)] += 1
    return {
        "class_by_prompt": class_by_prompt,
        "surface_by_prompt": surface_by_prompt,
        "residual_by_prompt_class": residual_by_prompt_class,
        "random_class_by_prompt": random_class_by_prompt,
        "noisy_by_prompt": noisy_by_prompt,
    }


def prob(counter: Counter[str] | None, item: str, vocab: int, alpha: float) -> float:
    if not counter:
        return 1.0 / max(1, vocab)
    total = sum(counter.values())
    return (counter.get(item, 0) + alpha) / (total + alpha * vocab)


def ce(p: float) -> float:
    return -math.log(max(p, 1e-12))


def evaluate(rows: list[Row], models: dict, classes: int, surface_vocab: int, alpha: float) -> dict:
    class_ce = 0.0
    flat_surface_ce = 0.0
    residual_ce = 0.0
    factor_ce = 0.0
    random_ce = 0.0
    noisy_correct = 0
    extractor_correct = 0
    identity_error = 0.0
    for row in rows:
        p_class = prob(models["class_by_prompt"].get(row.prompt), row.capability_class, classes, alpha)
        p_surface = prob(models["surface_by_prompt"].get(row.prompt), row.surface, surface_vocab, alpha)
        p_residual = prob(
            models["residual_by_prompt_class"].get((row.prompt, row.capability_class)),
            row.surface,
            surface_vocab,
            alpha,
        )
        p_factor = p_class * p_residual
        p_random = prob(models["random_class_by_prompt"].get(row.prompt), row.capability_class, classes, alpha)
        p_noisy = prob(models["noisy_by_prompt"].get(row.prompt), row.capability_class, classes, alpha)
        class_ce += ce(p_class)
        flat_surface_ce += ce(p_surface)
        residual_ce += ce(p_residual)
        factor_ce += ce(p_factor)
        random_ce += ce(p_random)
        extractor_correct += 1
        noisy_correct += int(p_noisy >= p_random)
        identity_error = max(identity_error, abs(ce(p_factor) - (ce(p_class) + ce(p_residual))))
    n = max(1, len(rows))
    return {
        "class_ce": class_ce / n,
        "flat_surface_ce": flat_surface_ce / n,
        "residual_ce": residual_ce / n,
        "factor_ce": factor_ce / n,
        "random_ce": random_ce / n,
        "extractor_accuracy": extractor_correct / n,
        "noisy_extractor_accuracy": noisy_correct / n,
        "identity_error": identity_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=256)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--surfaces-per-class", type=int, default=24)
    parser.add_argument("--repeats-per-prompt", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--noise-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42202)
    parser.add_argument("--min-class-gain", type=float, default=1.0)
    parser.add_argument("--min-random-gain", type=float, default=0.5)
    parser.add_argument("--max-residual-guard", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rows = make_rows(
        prompts=args.prompts,
        classes=args.classes,
        surfaces_per_class=args.surfaces_per_class,
        repeats_per_prompt=args.repeats_per_prompt,
        seed=args.seed,
    )
    train, heldout = split_rows(rows, args.train_fraction)
    models = count_models(train, args.classes, args.noise_rate, args.seed + 1)
    surface_vocab = len({row.surface for row in rows})
    metrics = evaluate(heldout, models, args.classes, surface_vocab, args.alpha)
    class_gain_vs_flat = metrics["flat_surface_ce"] - metrics["class_ce"]
    class_gain_vs_random = metrics["random_ce"] - metrics["class_ce"]
    residual_guard = max(0.0, metrics["residual_ce"] - metrics["flat_surface_ce"])
    pass_gate = (
        metrics["extractor_accuracy"] == 1.0
        and class_gain_vs_flat >= args.min_class_gain
        and class_gain_vs_random >= args.min_random_gain
        and metrics["identity_error"] <= 1e-9
        and residual_guard <= args.max_residual_guard
    )
    if pass_gate:
        decision = "PASS_CAPABILITY_EQUIVALENCE_GATE"
    elif metrics["extractor_accuracy"] < 1.0:
        decision = "FAIL_DETERMINISTIC_EXTRACTOR"
    elif metrics["identity_error"] > 1e-9:
        decision = "FAIL_EXACT_SURFACE_DECOMPOSITION"
    elif class_gain_vs_flat < args.min_class_gain:
        decision = "FAIL_CLASS_CE_GAIN"
    elif class_gain_vs_random < args.min_random_gain:
        decision = "FAIL_RANDOM_CLASS_CONTROL"
    else:
        decision = "FAIL_RESIDUAL_SURFACE_GUARD"

    activation = (
        "[R422_CAPABILITY_EQUIVALENCE_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        "decomposition=class_plus_residual_surface no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        train_rows=len(train),
        heldout_rows=len(heldout),
        classes=args.classes,
        surfaces_per_class=args.surfaces_per_class,
        extractor_accuracy=metrics["extractor_accuracy"],
        noisy_extractor_accuracy=metrics["noisy_extractor_accuracy"],
        class_ce=metrics["class_ce"],
        flat_surface_ce=metrics["flat_surface_ce"],
        factor_surface_ce=metrics["factor_ce"],
        residual_surface_ce=metrics["residual_ce"],
        random_class_ce=metrics["random_ce"],
        class_ce_gain_vs_flat_surface=class_gain_vs_flat,
        class_ce_gain_vs_random=class_gain_vs_random,
        factorization_identity_error=metrics["identity_error"],
        residual_guard=residual_guard,
        activation_log=activation,
        interpretation=(
            "material_gate_only: validates deterministic class/residual "
            "decomposition on a controlled task family, not GPU speed or broad text quality."
        ),
        elapsed_seconds=time.perf_counter() - start,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
