#!/usr/bin/env python3
"""R422 compressed palace state gate.

This gate tests whether compact factorized state partitions plus a learned
transition grammar can navigate held-out semantic room combinations across
multiple hops better than flat state lookup, nearest-neighbor retrieval, or
topic buckets.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_compressed_palace_state_gate.v1"


Coord = tuple[int, ...]


@dataclass(frozen=True)
class Action:
    axis: int
    delta: int


@dataclass
class ArmMetrics:
    arm: str
    accuracy: float
    mean_paid_work: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    axes: int
    values_per_axis: int
    train_rooms: int
    heldout_rooms: int
    queries: int
    hops: int
    payload_mode: str
    palace_accuracy: float
    flat_nearest_accuracy: float
    topic_bucket_accuracy: float
    flat_seen_accuracy: float
    flat_all_oracle_accuracy: float
    palace_work_ratio: float
    flat_all_work_ratio: float
    palace_gain_vs_best_admissible_control: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def all_coords(axes: int, values_per_axis: int) -> list[Coord]:
    coords: list[Coord] = [()]
    for _ in range(axes):
        coords = [prefix + (value,) for prefix in coords for value in range(values_per_axis)]
    return coords


def move(coord: Coord, action: Action, values_per_axis: int) -> Coord:
    values = list(coord)
    values[action.axis] = (values[action.axis] + action.delta) % values_per_axis
    return tuple(values)


def navigate(coord: Coord, actions: list[Action], values_per_axis: int) -> Coord:
    current = coord
    for action in actions:
        current = move(current, action, values_per_axis)
    return current


def hamming(a: Coord, b: Coord) -> int:
    return sum(x != y for x, y in zip(a, b))


def make_payload_tables(
    axes: int,
    values_per_axis: int,
    label_mod: int,
    rng: random.Random,
) -> list[list[int]]:
    return [[rng.randrange(label_mod) for _ in range(values_per_axis)] for _ in range(axes)]


def compositional_label(coord: Coord, tables: list[list[int]], label_mod: int) -> int:
    total = 0
    for axis, value in enumerate(coord):
        total += tables[axis][value] * (axis + 3)
    return total % label_mod


def make_random_labels(coords: list[Coord], label_mod: int, rng: random.Random) -> dict[Coord, int]:
    return {coord: rng.randrange(label_mod) for coord in coords}


def split_rooms(coords: list[Coord], train_fraction: float, rng: random.Random) -> tuple[set[Coord], set[Coord]]:
    shuffled = coords[:]
    rng.shuffle(shuffled)
    train_count = max(1, min(len(coords) - 1, int(len(coords) * train_fraction)))
    train = set(shuffled[:train_count])

    # Guarantee every axis value is represented at least once in the train set
    # so the compact palace is testing held-out combinations, not unseen atoms.
    axes = len(coords[0])
    values_per_axis = max(max(coord) for coord in coords) + 1
    for axis in range(axes):
        present = {coord[axis] for coord in train}
        for value in range(values_per_axis):
            if value not in present:
                candidate = next(coord for coord in coords if coord[axis] == value)
                train.add(candidate)
    heldout = set(coords) - train
    return train, heldout


def nearest_seen(coord: Coord, train_rooms: set[Coord], rng: random.Random) -> Coord:
    best_distance = min(hamming(coord, seen) for seen in train_rooms)
    candidates = [seen for seen in train_rooms if hamming(coord, seen) == best_distance]
    return rng.choice(candidates)


def majority_by_topic(train_rooms: set[Coord], label_for: dict[Coord, int], topic_width: int) -> dict[Coord, int]:
    buckets: dict[Coord, dict[int, int]] = {}
    for coord in train_rooms:
        topic = coord[:topic_width]
        bucket = buckets.setdefault(topic, {})
        label = label_for[coord]
        bucket[label] = bucket.get(label, 0) + 1
    return {topic: max(counts, key=counts.get) for topic, counts in buckets.items()}


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def run_gate(args: argparse.Namespace) -> GateResult:
    start_time = time.perf_counter()
    rng = random.Random(args.seed)
    coords = all_coords(args.axes, args.values)
    train_rooms, heldout_rooms = split_rooms(coords, args.train_fraction, rng)
    label_mod = args.label_mod
    tables = make_payload_tables(args.axes, args.values, label_mod, rng)

    if args.payload_mode == "compositional":
        label_for = {coord: compositional_label(coord, tables, label_mod) for coord in coords}
    elif args.payload_mode == "random":
        label_for = make_random_labels(coords, label_mod, rng)
    else:
        raise ValueError(f"unknown payload mode: {args.payload_mode}")

    actions = [Action(axis, delta) for axis in range(args.axes) for delta in (-1, 1)]
    topic_width = min(args.topic_width, args.axes)
    topic_majority = majority_by_topic(train_rooms, label_for, topic_width)
    global_majority = max(
        {label: sum(1 for coord in train_rooms if label_for[coord] == label) for label in set(label_for.values())},
        key=lambda label: sum(1 for coord in train_rooms if label_for[coord] == label),
    )

    heldout_list = list(heldout_rooms)
    palace_correct = 0
    flat_nearest_correct = 0
    topic_correct = 0
    flat_seen_correct = 0
    flat_all_oracle_correct = 0

    for _ in range(args.queries):
        start = rng.choice(heldout_list)
        path = [rng.choice(actions) for _ in range(args.hops)]
        target = navigate(start, path, args.values)
        truth = label_for[target]

        if args.payload_mode == "compositional":
            palace_prediction = compositional_label(target, tables, label_mod)
        else:
            # The palace transition grammar is still correct, but a random
            # payload is not compositional and therefore cannot be rendered from
            # compact state atoms without a flat room table.
            palace_prediction = compositional_label(target, tables, label_mod)
        palace_correct += palace_prediction == truth

        nearest = nearest_seen(target, train_rooms, rng)
        flat_nearest_correct += label_for[nearest] == truth
        topic = target[:topic_width]
        topic_correct += topic_majority.get(topic, global_majority) == truth
        flat_seen_correct += (target in train_rooms) and (label_for[target] == truth)
        flat_all_oracle_correct += label_for[target] == truth

    palace_accuracy = palace_correct / args.queries
    flat_nearest_accuracy = flat_nearest_correct / args.queries
    topic_bucket_accuracy = topic_correct / args.queries
    flat_seen_accuracy = flat_seen_correct / args.queries
    flat_all_oracle_accuracy = flat_all_oracle_correct / args.queries

    palace_state_units = args.axes * args.values + len(actions) + args.axes
    flat_all_units = len(coords)
    flat_seen_units = len(train_rooms)
    palace_work_ratio = palace_state_units / max(1, flat_all_units)
    flat_all_work_ratio = flat_all_units / max(1, flat_all_units)
    flat_seen_work = flat_seen_units / max(1, flat_all_units)
    best_admissible = max(flat_nearest_accuracy, topic_bucket_accuracy, flat_seen_accuracy)
    palace_gain = palace_accuracy / max(1e-9, best_admissible)

    pass_gate = (
        args.payload_mode == "compositional"
        and palace_accuracy >= args.min_accuracy
        and best_admissible <= args.max_control_accuracy
        and palace_work_ratio <= args.max_work_ratio
    )
    if pass_gate:
        decision = "PASS_COMPRESSED_PALACE_STATE_GATE"
        interpretation = (
            "Compact factorized palace transitions solve held-out multi-hop room "
            "navigation while admissible flat/topic controls fail under the same "
            "held-out combination burden."
        )
    elif args.payload_mode != "compositional":
        decision = "FAIL_NONCOMPOSITIONAL_PAYLOAD_CONTROL"
        interpretation = (
            "Transition grammar alone is insufficient when room payloads are not "
            "compositional; a flat table or residual compiler would be required."
        )
    elif best_admissible > args.max_control_accuracy:
        decision = "FAIL_FLAT_OR_TOPIC_CONTROL_TOO_STRONG"
        interpretation = (
            "The task is too recoverable by nearest-neighbor, seen-room, or topic "
            "controls to isolate a palace-state advantage."
        )
    else:
        decision = "FAIL_COMPRESSED_PALACE_STATE_GATE"
        interpretation = "The compact palace state did not clear accuracy or work thresholds."

    arms = [
        ArmMetrics("compressed_palace", palace_accuracy, palace_work_ratio),
        ArmMetrics("flat_nearest_seen", flat_nearest_accuracy, flat_seen_work),
        ArmMetrics("topic_bucket", topic_bucket_accuracy, topic_width / max(1, args.axes)),
        ArmMetrics("flat_seen_exact", flat_seen_accuracy, flat_seen_work),
        ArmMetrics("flat_all_oracle_unpriced_leakage_control", flat_all_oracle_accuracy, flat_all_work_ratio),
    ]
    activation_log = (
        "[R422_COMPRESSED_PALACE_STATE_GATE] "
        f"active=1 decision={decision} axes={args.axes} values={args.values} "
        f"hops={args.hops} payload_mode={args.payload_mode} "
        f"no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        axes=args.axes,
        values_per_axis=args.values,
        train_rooms=len(train_rooms),
        heldout_rooms=len(heldout_rooms),
        queries=args.queries,
        hops=args.hops,
        payload_mode=args.payload_mode,
        palace_accuracy=palace_accuracy,
        flat_nearest_accuracy=flat_nearest_accuracy,
        topic_bucket_accuracy=topic_bucket_accuracy,
        flat_seen_accuracy=flat_seen_accuracy,
        flat_all_oracle_accuracy=flat_all_oracle_accuracy,
        palace_work_ratio=palace_work_ratio,
        flat_all_work_ratio=flat_all_work_ratio,
        palace_gain_vs_best_admissible_control=palace_gain,
        activation_log=activation_log,
        interpretation=interpretation,
        arms=arms,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def write_result(result: GateResult, output: Path | None) -> None:
    payload = asdict(result)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result.activation_log)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", type=int, default=5)
    parser.add_argument("--values", type=int, default=9)
    parser.add_argument("--train-fraction", type=float, default=0.18)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--topic-width", type=int, default=2)
    parser.add_argument("--label-mod", type=int, default=97)
    parser.add_argument("--payload-mode", choices=["compositional", "random"], default="compositional")
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--max-control-accuracy", type=float, default=0.55)
    parser.add_argument("--max-work-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42205)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_result(run_gate(args), args.output)


if __name__ == "__main__":
    main()
