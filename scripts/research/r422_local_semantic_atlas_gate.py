#!/usr/bin/env python3
"""R422 local semantic atlas gate.

Tests whether local coordinate charts and transition maps transfer relation
vectors across incompatible domains better than monolithic latent averaging,
topic/MoE-style labels, or retrieval-like local deltas.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_local_semantic_atlas_gate.v1"


@dataclass
class DomainMap:
    perm: list[int]
    signs: list[float]
    bias: list[float]


@dataclass
class ArmMetrics:
    arm: str
    accuracy: float
    mean_relative_error: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    domains: int
    dimension: int
    heldout_domains: int
    queries: int
    atlas_accuracy: float
    atlas_mean_relative_error: float
    monolithic_accuracy: float
    topic_label_accuracy: float
    retrieval_delta_accuracy: float
    atlas_priced_ratio: float
    atlas_gain_vs_best_control: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def make_domain_map(dimension: int, rng: random.Random, aligned: bool) -> DomainMap:
    perm = list(range(dimension))
    signs = [1.0] * dimension
    if not aligned:
        rng.shuffle(perm)
        signs = [rng.choice([-1.0, 1.0]) for _ in range(dimension)]
    bias = [rng.uniform(-0.5, 0.5) for _ in range(dimension)]
    return DomainMap(perm, signs, bias)


def to_local(z: list[float], mapping: DomainMap) -> list[float]:
    return [mapping.signs[i] * z[mapping.perm[i]] + mapping.bias[i] for i in range(len(z))]


def to_canonical(x: list[float], mapping: DomainMap) -> list[float]:
    z = [0.0] * len(x)
    for i, source in enumerate(mapping.perm):
        z[source] = mapping.signs[i] * (x[i] - mapping.bias[i])
    return z


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def mean_vec(vectors: list[list[float]]) -> list[float]:
    n = max(1, len(vectors))
    return [sum(vec[i] for vec in vectors) / n for i in range(len(vectors[0]))]


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def random_vec(dimension: int, rng: random.Random, scale: float = 1.0) -> list[float]:
    return [rng.gauss(0.0, scale) for _ in range(dimension)]


def relative_error(pred: list[float], truth: list[float], denom: float) -> float:
    return norm(sub(pred, truth)) / max(1e-9, denom)


def score(errors: list[float], threshold: float) -> ArmMetrics:
    return ArmMetrics(
        arm="",
        accuracy=sum(err <= threshold for err in errors) / max(1, len(errors)),
        mean_relative_error=sum(errors) / max(1, len(errors)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", type=int, default=24)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--support-per-train-domain", type=int, default=16)
    parser.add_argument("--queries", type=int, default=2048)
    parser.add_argument("--heldout-fraction", type=float, default=0.33)
    parser.add_argument("--topics", type=int, default=4)
    parser.add_argument("--aligned-charts", action="store_true")
    parser.add_argument("--atlas-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42202)
    parser.add_argument("--error-threshold", type=float, default=0.05)
    parser.add_argument("--min-atlas-accuracy", type=float, default=0.98)
    parser.add_argument("--max-control-accuracy", type=float, default=0.50)
    parser.add_argument("--max-priced-ratio", type=float, default=0.75)
    parser.add_argument("--min-gain-vs-control", type=float, default=1.50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    maps = [make_domain_map(args.dimension, rng, args.aligned_charts) for _ in range(args.domains)]
    relation = random_vec(args.dimension, rng, scale=0.35)
    relation_norm = norm(relation)

    heldout_count = max(1, int(args.domains * args.heldout_fraction))
    train_domains = list(range(args.domains - heldout_count))
    heldout_domains = list(range(args.domains - heldout_count, args.domains))

    canonical_deltas: list[list[float]] = []
    raw_deltas: list[list[float]] = []
    raw_deltas_by_topic: dict[int, list[list[float]]] = {topic: [] for topic in range(args.topics)}
    support_sources: list[tuple[list[float], list[float]]] = []
    for domain in train_domains:
        mapping = maps[domain]
        for _ in range(args.support_per_train_domain):
            z = random_vec(args.dimension, rng)
            source = to_local(z, mapping)
            target = to_local(add(z, relation), mapping)
            raw_delta = sub(target, source)
            canonical_delta = sub(to_canonical(target, mapping), to_canonical(source, mapping))
            canonical_deltas.append(canonical_delta)
            raw_deltas.append(raw_delta)
            raw_deltas_by_topic[domain % args.topics].append(raw_delta)
            support_sources.append((source, raw_delta))

    atlas_relation = mean_vec(canonical_deltas)
    if args.atlas_noise:
        noise_scale = args.atlas_noise * max(1e-9, relation_norm) / math.sqrt(args.dimension)
        atlas_relation = [value + rng.gauss(0.0, noise_scale) for value in atlas_relation]
    monolithic_delta = mean_vec(raw_deltas)
    topic_delta = {
        topic: mean_vec(vectors) if vectors else monolithic_delta
        for topic, vectors in raw_deltas_by_topic.items()
    }

    errors: dict[str, list[float]] = {
        "local_semantic_atlas": [],
        "monolithic_latent_average": [],
        "topic_label_route": [],
        "retrieval_delta": [],
    }
    for _ in range(args.queries):
        domain = rng.choice(heldout_domains)
        mapping = maps[domain]
        z = random_vec(args.dimension, rng)
        source = to_local(z, mapping)
        truth = to_local(add(z, relation), mapping)
        denom = max(1e-9, relation_norm)

        atlas_pred = to_local(add(to_canonical(source, mapping), atlas_relation), mapping)
        mono_pred = add(source, monolithic_delta)
        topic_pred = add(source, topic_delta[domain % args.topics])
        nearest_source, nearest_delta = min(
            support_sources,
            key=lambda item: norm(sub(item[0], source)),
        )
        del nearest_source
        retrieval_pred = add(source, nearest_delta)

        errors["local_semantic_atlas"].append(relative_error(atlas_pred, truth, denom))
        errors["monolithic_latent_average"].append(relative_error(mono_pred, truth, denom))
        errors["topic_label_route"].append(relative_error(topic_pred, truth, denom))
        errors["retrieval_delta"].append(relative_error(retrieval_pred, truth, denom))

    arms: list[ArmMetrics] = []
    for name, err in errors.items():
        metrics = score(err, args.error_threshold)
        metrics.arm = name
        arms.append(metrics)

    atlas = next(arm for arm in arms if arm.arm == "local_semantic_atlas")
    controls = [arm for arm in arms if arm.arm != "local_semantic_atlas"]
    best_control_acc = max(arm.accuracy for arm in controls)
    best_control_error = min(arm.mean_relative_error for arm in controls)
    atlas_gain = best_control_error / max(1e-9, atlas.mean_relative_error)

    atlas_cost = len(train_domains) * args.support_per_train_domain + args.domains * args.dimension
    per_domain_cost = args.domains * args.support_per_train_domain * args.dimension
    priced_ratio = atlas_cost / max(1.0, per_domain_cost)

    pass_gate = (
        atlas.accuracy >= args.min_atlas_accuracy
        and best_control_acc <= args.max_control_accuracy
        and priced_ratio <= args.max_priced_ratio
        and atlas_gain >= args.min_gain_vs_control
    )
    if pass_gate:
        decision = "PASS_LOCAL_SEMANTIC_ATLAS_GATE"
    elif atlas.accuracy < args.min_atlas_accuracy:
        decision = "FAIL_ATLAS_TRANSFER_ACCURACY"
    elif best_control_acc > args.max_control_accuracy:
        decision = "FAIL_TOPIC_OR_MONOLITHIC_CONTROL"
    elif priced_ratio > args.max_priced_ratio:
        decision = "FAIL_ATLAS_PRICING"
    else:
        decision = "FAIL_TRANSFER_GAIN"

    activation = (
        "[R422_LOCAL_SEMANTIC_ATLAS_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        f"aligned_charts={int(args.aligned_charts)} atlas_noise={args.atlas_noise:.6f} "
        "no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        domains=args.domains,
        dimension=args.dimension,
        heldout_domains=len(heldout_domains),
        queries=args.queries,
        atlas_accuracy=atlas.accuracy,
        atlas_mean_relative_error=atlas.mean_relative_error,
        monolithic_accuracy=next(arm.accuracy for arm in arms if arm.arm == "monolithic_latent_average"),
        topic_label_accuracy=next(arm.accuracy for arm in arms if arm.arm == "topic_label_route"),
        retrieval_delta_accuracy=next(arm.accuracy for arm in arms if arm.arm == "retrieval_delta"),
        atlas_priced_ratio=priced_ratio,
        atlas_gain_vs_best_control=atlas_gain,
        activation_log=activation,
        interpretation=(
            "controlled_structural_gate_only: tests chart/transition-map relation transfer "
            "against topic/MoE and monolithic controls; not a production v12 path or speed claim."
        ),
        arms=arms,
        elapsed_seconds=time.perf_counter() - start,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(activation)
    print(
        json.dumps(
            {
                "decision": decision,
                "pass_gate": pass_gate,
                "atlas_accuracy": atlas.accuracy,
                "atlas_mean_relative_error": atlas.mean_relative_error,
                "monolithic_accuracy": payload.monolithic_accuracy,
                "topic_label_accuracy": payload.topic_label_accuracy,
                "retrieval_delta_accuracy": payload.retrieval_delta_accuracy,
                "atlas_priced_ratio": priced_ratio,
                "atlas_gain_vs_best_control": atlas_gain,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
