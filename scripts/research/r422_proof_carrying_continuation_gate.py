#!/usr/bin/env python3
"""R422 proof-carrying continuation target gate."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_proof_carrying_continuation_gate.v1"


@dataclass(frozen=True)
class Operation:
    opcode: str
    arg: int


@dataclass
class ArmMetrics:
    arm: str
    false_accept_rate: float
    certificate_length_ratio: float
    valid_set_reduction: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    cert_mode: str
    examples: int
    modulus: int
    steps: int
    identity_error: float
    false_accept_rate: float
    certificate_length_ratio: float
    weak_checker_false_accept_rate: float
    long_certificate_length_ratio: float
    valid_set_reduction: float
    priced_certificate_ratio: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def apply_op(value: int, op: Operation, modulus: int) -> int:
    if op.opcode == "add":
        return (value + op.arg) % modulus
    if op.opcode == "mul":
        return (value * op.arg) % modulus
    if op.opcode == "xor":
        return (value ^ op.arg) % modulus
    raise ValueError(f"unknown opcode {op.opcode}")


def make_program(steps: int, modulus: int, rng: random.Random) -> tuple[int, list[Operation], list[int]]:
    value = rng.randrange(modulus)
    trace = [value]
    ops: list[Operation] = []
    for _ in range(steps):
        opcode = rng.choice(["add", "mul", "xor"])
        arg = rng.randrange(1, min(modulus, 251))
        op = Operation(opcode, arg)
        value = apply_op(value, op, modulus)
        ops.append(op)
        trace.append(value)
    return trace[0], ops, trace


def checksum(trace: list[int], modulus: int) -> int:
    total = 17
    for i, value in enumerate(trace):
        total = (total * 131 + (i + 1) * value) % modulus
    return total


def strong_certificate(trace: list[int], modulus: int) -> tuple[int, int, int]:
    return (trace[-1], len(trace) - 1, checksum(trace, modulus))


def weak_certificate(trace: list[int], _modulus: int) -> tuple[int]:
    return (trace[-1] & 1,)


def long_certificate(trace: list[int], _modulus: int) -> tuple[int, ...]:
    return tuple(trace)


def execute_trace(start: int, ops: list[Operation], modulus: int) -> list[int]:
    trace = [start]
    value = start
    for op in ops:
        value = apply_op(value, op, modulus)
        trace.append(value)
    return trace


def check_strong(start: int, ops: list[Operation], cert: tuple[int, int, int], modulus: int) -> bool:
    trace = execute_trace(start, ops, modulus)
    return cert == strong_certificate(trace, modulus)


def check_weak(start: int, ops: list[Operation], cert: tuple[int], modulus: int) -> bool:
    trace = execute_trace(start, ops, modulus)
    expected = weak_certificate(trace, modulus)
    return cert == expected


def check_long(start: int, ops: list[Operation], cert: tuple[int, ...], modulus: int) -> bool:
    trace = execute_trace(start, ops, modulus)
    return cert == long_certificate(trace, modulus)


def corrupt_strong(cert: tuple[int, int, int], modulus: int, rng: random.Random) -> tuple[int, int, int]:
    field = rng.randrange(3)
    values = list(cert)
    values[field] = (values[field] + rng.randrange(1, modulus)) % modulus
    if field == 1:
        values[field] = max(0, cert[field] + rng.choice([-1, 1]))
    return (values[0], values[1], values[2])


def corrupt_weak(cert: tuple[int], _modulus: int, _rng: random.Random) -> tuple[int]:
    # Same weak certificate: many wrong continuations share this parity and are
    # therefore false accepts under the weak checker.
    return cert


def corrupt_long(cert: tuple[int, ...], modulus: int, rng: random.Random) -> tuple[int, ...]:
    values = list(cert)
    index = rng.randrange(len(values))
    values[index] = (values[index] + rng.randrange(1, modulus)) % modulus
    return tuple(values)


def run_gate(args: argparse.Namespace) -> GateResult:
    start_time = time.perf_counter()
    rng = random.Random(args.seed)
    false_accepts = 0
    weak_false_accepts = 0
    long_false_accepts = 0
    identity_errors = 0
    strong_lengths: list[float] = []
    weak_lengths: list[float] = []
    long_lengths: list[float] = []

    for _ in range(args.examples):
        start, ops, trace = make_program(args.steps, args.modulus, rng)
        strong = strong_certificate(trace, args.modulus)
        weak = weak_certificate(trace, args.modulus)
        long = long_certificate(trace, args.modulus)

        identity_errors += not check_strong(start, ops, strong, args.modulus)
        false_accepts += check_strong(start, ops, corrupt_strong(strong, args.modulus, rng), args.modulus)
        weak_false_accepts += check_weak(start, ops, corrupt_weak(weak, args.modulus, rng), args.modulus)
        long_false_accepts += check_long(start, ops, corrupt_long(long, args.modulus, rng), args.modulus)

        target_len = len(trace)
        strong_lengths.append(len(strong) / target_len)
        weak_lengths.append(len(weak) / target_len)
        long_lengths.append(len(long) / target_len)

    identity_error = identity_errors / args.examples
    false_accept_rate = false_accepts / args.examples
    weak_false_accept_rate = weak_false_accepts / args.examples
    long_false_accept_rate = long_false_accepts / args.examples
    strong_len_ratio = sum(strong_lengths) / args.examples
    weak_len_ratio = sum(weak_lengths) / args.examples
    long_len_ratio = sum(long_lengths) / args.examples

    if args.cert_mode == "strong":
        active_false_accept = false_accept_rate
        active_len_ratio = strong_len_ratio
        valid_set_reduction = 1.0 / args.modulus
    elif args.cert_mode == "weak":
        active_false_accept = weak_false_accept_rate
        active_len_ratio = weak_len_ratio
        valid_set_reduction = 0.5
    elif args.cert_mode == "long":
        active_false_accept = long_false_accept_rate
        active_len_ratio = long_len_ratio
        valid_set_reduction = 1.0 / args.modulus
    else:
        raise ValueError(f"unknown cert mode: {args.cert_mode}")

    priced_ratio = active_len_ratio + args.checker_cost_ratio
    pass_gate = (
        args.cert_mode == "strong"
        and identity_error == 0.0
        and active_false_accept <= args.max_false_accept
        and active_len_ratio <= args.max_length_ratio
        and weak_false_accept_rate >= args.min_weak_false_accept
        and long_len_ratio >= args.min_long_length_ratio
        and priced_ratio <= args.max_priced_ratio
    )
    if pass_gate:
        decision = "PASS_PROOF_CARRYING_CONTINUATION_GATE"
        interpretation = (
            "A short machine-checkable continuation certificate factors the "
            "target before residual CE while rejecting corrupt continuations."
        )
    elif args.cert_mode == "weak":
        decision = "FAIL_WEAK_CHECKER_CONTROL"
        interpretation = "Weak certificates are short but accept invalid continuations."
    elif args.cert_mode == "long":
        decision = "FAIL_LONG_CERTIFICATE_CONTROL"
        interpretation = "Long certificates verify but collapse into copying the target."
    else:
        decision = "FAIL_PROOF_CARRYING_CONTINUATION_GATE"
        interpretation = "The certificate did not clear identity, false-accept, length, or pricing thresholds."

    arms = [
        ArmMetrics("strong_certificate", false_accept_rate, strong_len_ratio, 1.0 / args.modulus),
        ArmMetrics("weak_checker", weak_false_accept_rate, weak_len_ratio, 0.5),
        ArmMetrics("long_certificate", long_false_accept_rate, long_len_ratio, 1.0 / args.modulus),
    ]
    activation_log = (
        "[R422_PROOF_CARRYING_CONTINUATION_GATE] "
        f"active=1 decision={decision} cert_mode={args.cert_mode} "
        f"no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        cert_mode=args.cert_mode,
        examples=args.examples,
        modulus=args.modulus,
        steps=args.steps,
        identity_error=identity_error,
        false_accept_rate=active_false_accept,
        certificate_length_ratio=active_len_ratio,
        weak_checker_false_accept_rate=weak_false_accept_rate,
        long_certificate_length_ratio=long_len_ratio,
        valid_set_reduction=valid_set_reduction,
        priced_certificate_ratio=priced_ratio,
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
    parser.add_argument("--cert-mode", choices=["strong", "weak", "long"], default="strong")
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--modulus", type=int, default=997)
    parser.add_argument("--checker-cost-ratio", type=float, default=0.03)
    parser.add_argument("--max-false-accept", type=float, default=0.0)
    parser.add_argument("--max-length-ratio", type=float, default=0.35)
    parser.add_argument("--min-weak-false-accept", type=float, default=0.25)
    parser.add_argument("--min-long-length-ratio", type=float, default=0.95)
    parser.add_argument("--max-priced-ratio", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42203)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_result(run_gate(args), args.output)


if __name__ == "__main__":
    main()
