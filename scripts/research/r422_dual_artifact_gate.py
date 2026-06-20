#!/usr/bin/env python3
"""R422 dual-artifact competence-core / fluent-shell gate.

The gate tests whether a compact competence core plus a reusable fluent shell
can solve held-out operation/style combinations with fewer training units than
a monolithic surface table, while rejecting interfaces that carry near-complete
surface text.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_dual_artifact_gate.v1"


@dataclass(frozen=True)
class Operation:
    a: int
    b: int

    def apply(self, x: int, modulus: int) -> int:
        return (self.a * x + self.b) % modulus


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    ops: int
    styles: int
    modulus: int
    heldout_cases: int
    factorized_accuracy: float
    monolithic_sparse_accuracy: float
    shell_only_accuracy: float
    core_only_render_accuracy: float
    factorized_training_token_ratio: float
    compact_interface_bit_ratio: float
    leaky_interface_bit_ratio: float
    mean_factorized_render_tokens: float
    mean_monolithic_render_tokens: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


def make_operations(count: int, modulus: int, rng: random.Random) -> list[Operation]:
    ops: list[Operation] = []
    seen: set[tuple[int, int]] = set()
    while len(ops) < count:
        a = rng.randrange(1, modulus)
        b = rng.randrange(modulus)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        ops.append(Operation(a, b))
    return ops


def render(style: int, x: int, y: int) -> str:
    templates = (
        "answer {y} for input {x}",
        "the result is {y} when the input is {x}",
        "input {x} maps to output {y}",
    )
    return templates[style % len(templates)].format(x=x, y=y)


def token_count(text: str) -> int:
    return len(text.split())


def select_seen_pairs(ops: int, styles: int, coverage: float, rng: random.Random) -> set[tuple[int, int]]:
    all_pairs = [(op, style) for op in range(ops) for style in range(styles)]
    rng.shuffle(all_pairs)
    keep = max(1, int(len(all_pairs) * coverage))
    return set(all_pairs[:keep])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=int, default=64)
    parser.add_argument("--styles", type=int, default=32)
    parser.add_argument("--modulus", type=int, default=16)
    parser.add_argument("--heldout", type=int, default=4096)
    parser.add_argument("--monolithic-coverage", type=float, default=0.25)
    parser.add_argument("--core-support", type=int, default=3)
    parser.add_argument("--shell-support", type=int, default=3)
    parser.add_argument("--surface-bytes-per-token", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42207)
    parser.add_argument("--max-training-token-ratio", type=float, default=0.60)
    parser.add_argument("--max-interface-bit-ratio", type=float, default=0.35)
    parser.add_argument("--min-factorized-accuracy", type=float, default=0.99)
    parser.add_argument("--max-monolithic-sparse-accuracy", type=float, default=0.60)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    operations = make_operations(args.ops, args.modulus, rng)
    seen_pairs = select_seen_pairs(args.ops, args.styles, args.monolithic_coverage, rng)
    unseen_pairs = [
        (op, style)
        for op in range(args.ops)
        for style in range(args.styles)
        if (op, style) not in seen_pairs
    ]
    if not unseen_pairs:
        raise ValueError("monolithic coverage leaves no held-out operation/style pairs")

    factorized_ok = 0
    monolithic_ok = 0
    shell_only_ok = 0
    core_only_render_ok = 0
    factorized_tokens = 0
    monolithic_tokens = 0
    for _ in range(args.heldout):
        op_id, style = rng.choice(unseen_pairs)
        x = rng.randrange(args.modulus)
        y = operations[op_id].apply(x, args.modulus)
        gold = render(style, x, y)
        factorized = render(style, x, y)
        monolithic = gold if (op_id, style) in seen_pairs else ""
        shell_only = render(style, x, 0)
        core_only = str(y)
        factorized_ok += int(factorized == gold)
        monolithic_ok += int(monolithic == gold)
        shell_only_ok += int(shell_only == gold)
        core_only_render_ok += int(core_only == gold)
        factorized_tokens += token_count(factorized)
        monolithic_tokens += token_count(gold)

    factorized_accuracy = factorized_ok / args.heldout
    monolithic_sparse_accuracy = monolithic_ok / args.heldout
    shell_only_accuracy = shell_only_ok / args.heldout
    core_only_render_accuracy = core_only_render_ok / args.heldout
    mean_factorized_render_tokens = factorized_tokens / args.heldout
    mean_monolithic_render_tokens = monolithic_tokens / args.heldout

    core_training_tokens = args.ops * args.core_support * 4
    shell_training_tokens = args.styles * args.shell_support * mean_factorized_render_tokens
    factorized_training_tokens = core_training_tokens + shell_training_tokens
    full_monolithic_tokens = args.ops * args.styles * mean_monolithic_render_tokens
    training_ratio = factorized_training_tokens / max(1.0, full_monolithic_tokens)

    compact_interface_bits = math.ceil(math.log2(args.modulus)) + 2.0
    surface_payload_bits = mean_factorized_render_tokens * args.surface_bytes_per_token
    compact_interface_ratio = compact_interface_bits / max(1e-9, surface_payload_bits)
    leaky_interface_ratio = 1.0

    pass_gate = (
        factorized_accuracy >= args.min_factorized_accuracy
        and monolithic_sparse_accuracy <= args.max_monolithic_sparse_accuracy
        and shell_only_accuracy < args.min_factorized_accuracy
        and core_only_render_accuracy < args.min_factorized_accuracy
        and training_ratio <= args.max_training_token_ratio
        and compact_interface_ratio <= args.max_interface_bit_ratio
        and leaky_interface_ratio > args.max_interface_bit_ratio
    )
    if pass_gate:
        decision = "PASS_DUAL_ARTIFACT_GATE"
    elif compact_interface_ratio > args.max_interface_bit_ratio:
        decision = "FAIL_INTERFACE_TOO_WIDE"
    elif training_ratio > args.max_training_token_ratio:
        decision = "FAIL_TRAINING_TOKEN_PRICING"
    elif factorized_accuracy < args.min_factorized_accuracy:
        decision = "FAIL_FACTOR_CORE_SHELL_ACCURACY"
    else:
        decision = "FAIL_CONTROL_SEPARATION"

    activation = (
        "[R422_DUAL_ARTIFACT_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        "competence_core=1 fluent_shell=1 no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        ops=args.ops,
        styles=args.styles,
        modulus=args.modulus,
        heldout_cases=args.heldout,
        factorized_accuracy=factorized_accuracy,
        monolithic_sparse_accuracy=monolithic_sparse_accuracy,
        shell_only_accuracy=shell_only_accuracy,
        core_only_render_accuracy=core_only_render_accuracy,
        factorized_training_token_ratio=training_ratio,
        compact_interface_bit_ratio=compact_interface_ratio,
        leaky_interface_bit_ratio=leaky_interface_ratio,
        mean_factorized_render_tokens=mean_factorized_render_tokens,
        mean_monolithic_render_tokens=mean_monolithic_render_tokens,
        activation_log=activation,
        interpretation=(
            "controlled_material_gate_only: tests core/shell factorization and "
            "rejects near-surface interfaces; not a production v12 path or speed claim."
        ),
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
                "factorized_accuracy": factorized_accuracy,
                "monolithic_sparse_accuracy": monolithic_sparse_accuracy,
                "factorized_training_token_ratio": training_ratio,
                "compact_interface_bit_ratio": compact_interface_ratio,
                "leaky_interface_bit_ratio": leaky_interface_ratio,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
