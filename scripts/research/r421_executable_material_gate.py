#!/usr/bin/env python3
"""R421 Constraint/Witness/Syndrome executable material gate.

This CPU-only gate tests whether causal, non-leaking witness material can expose
structure before lexical residuals while preserving an explicit original-CE
charge.

The witness is intentionally narrow: it is computed only from prefix state
(previous byte classes, column bucket, bracket depth, and quote state). It
predicts a target byte class/syndrome, while the token residual is still charged
inside that class. Removed-witness and shuffled-witness controls test whether
the witness carries real causal structure rather than target leakage.

It does not modify production training and does not measure GPU speed.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
BYTE_VOCAB = 256
SCHEMA = "r421_executable_material_gate.v1"


@dataclass
class GateMetrics:
    token_unigram_ce: float
    class_unigram_ce: float
    witness_class_ce: float
    shuffled_class_ce: float
    class_residual_ce: float
    witness_factor_ce: float
    removed_factor_ce: float
    shuffled_factor_ce: float
    class_ce_gain_over_removed: float
    class_ce_gain_over_shuffled: float
    factor_ce_gain_over_removed: float
    factor_ce_gain_over_shuffled: float
    priced_work_ratio: float
    support_speedup_estimate: float
    exact_feature_token_purity: float
    eval_positions: int
    feature_count: int
    pass_gate: bool


def read_v12_tokens(path: Path, max_tokens: int | None) -> np.ndarray:
    data = path.read_bytes()
    offset = 0
    n_tokens = len(data) // 2
    if len(data) >= DATASET_HEADER_BYTES:
        magic, version, header_tokens = struct.unpack_from("<IIQ", data, 0)
        if magic == DATASET_MAGIC:
            if version != 1:
                raise ValueError(f"unsupported dataset version: {version}")
            offset = DATASET_HEADER_BYTES
            n_tokens = int(header_tokens)
    available = (len(data) - offset) // 2
    n = min(n_tokens, available)
    if max_tokens is not None:
        n = min(n, max_tokens)
    if n <= 32:
        raise ValueError("dataset does not contain enough tokens")
    return np.frombuffer(data, dtype="<u2", count=n, offset=offset).astype(np.int64) % BYTE_VOCAB


def byte_class(token: int) -> int:
    if token in (9, 10, 13, 32):
        return 0
    if 48 <= token <= 57:
        return 1
    if 65 <= token <= 90:
        return 2
    if token in (97, 101, 105, 111, 117):
        return 3
    if 97 <= token <= 122:
        return 4
    if token in (40, 41, 91, 93, 123, 125, 60, 62):
        return 5
    if 33 <= token <= 126:
        return 6
    return 7


def update_prefix_state(token: int, state: dict[str, int]) -> None:
    if token in (10, 13):
        state["column"] = 0
    else:
        state["column"] += 1
    if token in (40, 91, 123, 60):
        state["depth"] = min(7, state["depth"] + 1)
    elif token in (41, 93, 125, 62):
        state["depth"] = max(0, state["depth"] - 1)
    if token in (34, 39, 96):
        state["quote"] ^= 1


def feature_stream(tokens: np.ndarray) -> tuple[list[tuple[int, int, int, int, int]], list[int], list[int]]:
    state = {"column": 0, "depth": 0, "quote": 0}
    prev1 = 0
    prev2 = 0
    features: list[tuple[int, int, int, int, int]] = []
    classes: list[int] = []
    raw_targets: list[int] = []
    for raw in tokens:
        token = int(raw)
        features.append((prev1, prev2, min(state["column"] // 4, 15), state["depth"], state["quote"]))
        classes.append(byte_class(token))
        raw_targets.append(token)
        prev2 = prev1
        prev1 = byte_class(token)
        update_prefix_state(token, state)
    return features, classes, raw_targets


def split_train_eval(
    features: list[tuple[int, int, int, int, int]],
    classes: list[int],
    targets: list[int],
    train_fraction: float,
) -> tuple[list, list, list, list, list, list]:
    split = max(16, min(len(classes) - 16, int(len(classes) * train_fraction)))
    return (
        features[:split],
        classes[:split],
        targets[:split],
        features[split:],
        classes[split:],
        targets[split:],
    )


def counter_table(keys: Iterable, values: Iterable[int]) -> dict[object, Counter[int]]:
    table: dict[object, Counter[int]] = defaultdict(Counter)
    for key, value in zip(keys, values):
        table[key][int(value)] += 1
    return table


def ce_from_table(
    keys: list,
    values: list[int],
    table: dict[object, Counter[int]],
    global_counts: Counter[int],
    n_classes: int,
    alpha: float,
) -> float:
    total = 0.0
    for key, value in zip(keys, values):
        counts = table.get(key)
        if counts:
            denom = sum(counts.values()) + alpha * n_classes
            prob = (counts.get(int(value), 0) + alpha) / denom
        else:
            denom = sum(global_counts.values()) + alpha * n_classes
            prob = (global_counts.get(int(value), 0) + alpha) / denom
        total -= math.log(max(prob, 1e-300))
    return total / max(len(values), 1)


def unigram_ce(values: list[int], train_counts: Counter[int], n_classes: int, alpha: float) -> float:
    denom = sum(train_counts.values()) + alpha * n_classes
    return sum(
        -math.log(max((train_counts.get(int(value), 0) + alpha) / denom, 1e-300))
        for value in values
    ) / max(len(values), 1)


def residual_ce_given_class(
    eval_classes: list[int],
    eval_targets: list[int],
    class_token_counts: dict[int, Counter[int]],
    global_token_counts: Counter[int],
    alpha: float,
) -> float:
    total = 0.0
    for cls, token in zip(eval_classes, eval_targets):
        counts = class_token_counts.get(int(cls))
        if counts:
            denom = sum(counts.values()) + alpha * BYTE_VOCAB
            prob = (counts.get(int(token), 0) + alpha) / denom
        else:
            denom = sum(global_token_counts.values()) + alpha * BYTE_VOCAB
            prob = (global_token_counts.get(int(token), 0) + alpha) / denom
        total -= math.log(max(prob, 1e-300))
    return total / max(len(eval_targets), 1)


def token_purity_by_feature(eval_features: list, eval_targets: list[int]) -> float:
    table = counter_table(eval_features, eval_targets)
    total = 0
    pure = 0
    for counts in table.values():
        n = sum(counts.values())
        total += n
        pure += max(counts.values())
    return pure / max(total, 1)


def shuffled_features(features: list, seed: int) -> list:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    rng.shuffle(indices)
    return [features[int(i)] for i in indices]


def run_gate(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    tokens = read_v12_tokens(args.dataset, args.max_tokens)
    features, classes, targets = feature_stream(tokens)
    train_f, train_c, train_t, eval_f, eval_c, eval_t = split_train_eval(
        features,
        classes,
        targets,
        args.train_fraction,
    )
    class_counts = Counter(train_c)
    token_counts = Counter(train_t)
    witness_table = counter_table(train_f, train_c)
    class_token_counts = counter_table(train_c, train_t)

    token_base = unigram_ce(eval_t, token_counts, BYTE_VOCAB, args.alpha)
    class_base = unigram_ce(eval_c, class_counts, args.class_count, args.alpha)
    witness_class = ce_from_table(eval_f, eval_c, witness_table, class_counts, args.class_count, args.alpha)
    shuffled_eval_f = shuffled_features(eval_f, args.seed + 17)
    shuffled_class = ce_from_table(
        shuffled_eval_f,
        eval_c,
        witness_table,
        class_counts,
        args.class_count,
        args.alpha,
    )
    residual = residual_ce_given_class(eval_c, eval_t, class_token_counts, token_counts, args.alpha)
    witness_factor = witness_class + residual
    removed_factor = class_base + residual
    shuffled_factor = shuffled_class + residual
    class_gain_removed = class_base - witness_class
    class_gain_shuffled = shuffled_class - witness_class
    factor_gain_removed = removed_factor - witness_factor
    factor_gain_shuffled = shuffled_factor - witness_factor
    avg_class_density = sum(class_counts[int(cls)] for cls in class_counts) / max(len(class_counts), 1)
    avg_class_density = min(1.0, (BYTE_VOCAB / args.class_count) / BYTE_VOCAB)
    priced_work_ratio = (
        args.class_head_work_share * (args.class_count / BYTE_VOCAB)
        + args.residual_work_share * avg_class_density
        + args.verifier_overhead
        + args.material_byte_overhead
    )
    speedup = 1.0 / max(priced_work_ratio, 1e-30)
    purity = token_purity_by_feature(eval_f, eval_t)
    pass_gate = (
        class_gain_removed >= args.min_class_ce_gain
        and class_gain_shuffled >= args.min_control_ce_gain
        and factor_gain_removed >= args.min_factor_ce_gain
        and factor_gain_shuffled >= args.min_control_ce_gain
        and priced_work_ratio <= args.max_priced_work_ratio
        and purity <= args.max_feature_token_purity
        and witness_factor <= token_base + args.max_factor_ce_over_token_base
    )
    metrics = GateMetrics(
        token_unigram_ce=float(token_base),
        class_unigram_ce=float(class_base),
        witness_class_ce=float(witness_class),
        shuffled_class_ce=float(shuffled_class),
        class_residual_ce=float(residual),
        witness_factor_ce=float(witness_factor),
        removed_factor_ce=float(removed_factor),
        shuffled_factor_ce=float(shuffled_factor),
        class_ce_gain_over_removed=float(class_gain_removed),
        class_ce_gain_over_shuffled=float(class_gain_shuffled),
        factor_ce_gain_over_removed=float(factor_gain_removed),
        factor_ce_gain_over_shuffled=float(factor_gain_shuffled),
        priced_work_ratio=float(priced_work_ratio),
        support_speedup_estimate=float(speedup),
        exact_feature_token_purity=float(purity),
        eval_positions=int(len(eval_c)),
        feature_count=int(len(set(eval_f))),
        pass_gate=bool(pass_gate),
    )
    if pass_gate:
        decision = "PASS_EXECUTABLE_MATERIAL_NONLEAKING_GATE"
    elif purity > args.max_feature_token_purity:
        decision = "FAIL_WITNESS_FEATURE_LOOKUP_LEAK"
    elif class_gain_shuffled < args.min_control_ce_gain or factor_gain_shuffled < args.min_control_ce_gain:
        decision = "FAIL_SHUFFLED_WITNESS_CONTROL"
    elif class_gain_removed < args.min_class_ce_gain or factor_gain_removed < args.min_factor_ce_gain:
        decision = "FAIL_WITNESS_DOES_NOT_REDUCE_CE"
    elif priced_work_ratio > args.max_priced_work_ratio:
        decision = "FAIL_MATERIAL_OVERHEAD_TOO_HIGH"
    else:
        decision = "PARK_EXECUTABLE_MATERIAL_BORDERLINE"
    return {
        "schema": SCHEMA,
        "gate": "r421_constraint_witness_syndrome_executable_material",
        "decision": decision,
        "dataset": str(args.dataset),
        "n_tokens": int(tokens.shape[0]),
        "train_fraction": float(args.train_fraction),
        "class_count": int(args.class_count),
        "seed": int(args.seed),
        "metrics": asdict(metrics),
        "thresholds": {
            "min_class_ce_gain": float(args.min_class_ce_gain),
            "min_factor_ce_gain": float(args.min_factor_ce_gain),
            "min_control_ce_gain": float(args.min_control_ce_gain),
            "max_priced_work_ratio": float(args.max_priced_work_ratio),
            "max_feature_token_purity": float(args.max_feature_token_purity),
            "max_factor_ce_over_token_base": float(args.max_factor_ce_over_token_base),
        },
        "causal_feature_contract": (
            "feature=(previous_byte_class, previous_previous_byte_class, "
            "prefix_column_bucket, prefix_bracket_depth, prefix_quote_state); "
            "no target byte or future byte is included"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "no_training_integration": True,
        "no_gpu": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/text_data.bin"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=421130)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--class-count", type=int, default=8)
    parser.add_argument("--class-head-work-share", type=float, default=0.25)
    parser.add_argument("--residual-work-share", type=float, default=0.75)
    parser.add_argument("--verifier-overhead", type=float, default=0.04)
    parser.add_argument("--material-byte-overhead", type=float, default=0.03)
    parser.add_argument("--min-class-ce-gain", type=float, default=0.05)
    parser.add_argument("--min-factor-ce-gain", type=float, default=0.05)
    parser.add_argument("--min-control-ce-gain", type=float, default=0.03)
    parser.add_argument("--max-priced-work-ratio", type=float, default=0.90)
    parser.add_argument("--max-feature-token-purity", type=float, default=0.75)
    parser.add_argument("--max-factor-ce-over-token-base", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gate(args)
    metrics = payload["metrics"]
    print(
        "[R421_EXECUTABLE_MATERIAL_GATE] "
        f"active=1 decision={payload['decision']} "
        f"class_gain={metrics['class_ce_gain_over_removed']:.6f} "
        f"control_gain={metrics['class_ce_gain_over_shuffled']:.6f} "
        f"factor_gain={metrics['factor_ce_gain_over_removed']:.6f} "
        f"priced_work_ratio={metrics['priced_work_ratio']:.6f} "
        f"purity={metrics['exact_feature_token_purity']:.6f} "
        "no_training_integration=1 no_gpu=1"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
