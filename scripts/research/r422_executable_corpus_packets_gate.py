#!/usr/bin/env python3
"""R422 TM03 executable corpus packets material gate.

Each packet contains code, config, input/output traces, and a natural label.
The gate tests held-out behavior prediction and mismatch diagnosis against
text-only, trace-nearest, hidden-lookup, and shuffled-code controls.

This is a CPU material/compiler gate only. It is not a v12/GPU training path and
does not measure training speed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_executable_corpus_packets_gate.v1"


@dataclass(frozen=True)
class Config:
    modulus: int
    threshold: int
    mul: int
    add: int
    jump: int
    energy_cost: int


@dataclass
class Packet:
    packet_id: int
    config: Config
    train_inputs: list[int]
    heldout_inputs: list[int]
    trace_steps: int
    natural_label: str


@dataclass
class GateResult:
    schema: str
    decision: str
    packets: int
    train_inputs_per_packet: int
    heldout_inputs_per_packet: int
    trace_steps: int
    executable_accuracy: float
    text_only_accuracy: float
    trace_nearest_accuracy: float
    hidden_lookup_accuracy: float
    shuffled_code_accuracy: float
    mismatch_detection_accuracy: float
    corrupted_trace_accept_rate: float
    packet_work_ratio: float
    executable_vs_best_control_gain: float
    activation_log: str
    compiler_material: bool
    no_gpu: bool
    no_training_integration: bool
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=256)
    parser.add_argument("--train-inputs", type=int, default=6)
    parser.add_argument("--heldout-inputs", type=int, default=8)
    parser.add_argument("--trace-steps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42203)
    parser.add_argument("--max-packet-work-ratio", type=float, default=0.45)
    parser.add_argument("--min-executable-accuracy", type=float, default=0.99)
    parser.add_argument("--max-best-control-accuracy", type=float, default=0.75)
    parser.add_argument("--min-mismatch-detection", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def step_value(x: int, cfg: Config) -> int:
    if x < cfg.threshold:
        return (x * cfg.mul + cfg.add) % cfg.modulus
    return (x + cfg.jump - cfg.energy_cost) % cfg.modulus


def run_program(x: int, cfg: Config, steps: int) -> list[int]:
    trace = [x]
    cur = x
    for _ in range(steps):
        cur = step_value(cur, cfg)
        trace.append(cur)
    return trace


def make_config(rng: random.Random) -> Config:
    modulus = rng.choice([53, 59, 61, 67, 71, 73, 79, 83])
    threshold = rng.randrange(8, modulus - 8)
    mul = rng.choice([2, 3, 5, 7, 11])
    add = rng.randrange(1, modulus)
    jump = rng.randrange(3, modulus)
    energy_cost = rng.randrange(1, 5)
    return Config(modulus, threshold, mul, add, jump, energy_cost)


def make_packet(rng: random.Random, packet_id: int, train_inputs: int, heldout_inputs: int, steps: int) -> Packet:
    cfg = make_config(rng)
    inputs = list(range(cfg.modulus))
    rng.shuffle(inputs)
    train = inputs[:train_inputs]
    heldout = inputs[train_inputs : train_inputs + heldout_inputs]
    label = f"piecewise modular transition m{cfg.modulus} t{cfg.threshold}"
    return Packet(packet_id, cfg, train, heldout, steps, label)


def nearest_trace_predict(x: int, packet: Packet) -> int:
    nearest = min(packet.train_inputs, key=lambda seen: abs(seen - x))
    return run_program(nearest, packet.config, packet.trace_steps)[-1]


def text_only_predict(packet: Packet) -> int:
    outputs = [run_program(x, packet.config, packet.trace_steps)[-1] for x in packet.train_inputs]
    return max(set(outputs), key=outputs.count)


def shuffled_config(cfg: Config) -> Config:
    return Config(
        modulus=cfg.modulus,
        threshold=max(1, cfg.modulus - cfg.threshold - 1),
        mul=cfg.add % cfg.modulus or 1,
        add=cfg.jump % cfg.modulus,
        jump=cfg.mul % cfg.modulus,
        energy_cost=cfg.energy_cost,
    )


def hidden_lookup_predict(x: int, packet: Packet) -> int | None:
    if x in packet.train_inputs:
        return run_program(x, packet.config, packet.trace_steps)[-1]
    return None


def diagnose_mismatch(packet: Packet, x: int, trace: list[int]) -> bool:
    return trace == run_program(x, packet.config, packet.trace_steps)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    packets = [
        make_packet(rng, i, args.train_inputs, args.heldout_inputs, args.trace_steps)
        for i in range(args.packets)
    ]

    total = 0
    executable_ok = 0
    text_ok = 0
    nearest_ok = 0
    lookup_ok = 0
    shuffled_ok = 0
    mismatch_ok = 0
    corrupted_accept = 0
    packet_work = 0.0
    control_work = 0.0
    corrupt_rng = random.Random(args.seed + 101)

    for packet in packets:
        cfg = packet.config
        bad_cfg = shuffled_config(cfg)
        packet_parse_cost = 6.0 + args.train_inputs * args.trace_steps * 0.25
        for x in packet.heldout_inputs:
            total += 1
            target = run_program(x, cfg, args.trace_steps)[-1]
            executable_ok += run_program(x, cfg, args.trace_steps)[-1] == target
            text_ok += text_only_predict(packet) == target
            nearest_ok += nearest_trace_predict(x, packet) == target
            lookup = hidden_lookup_predict(x, packet)
            lookup_ok += lookup == target
            shuffled_ok += run_program(x, bad_cfg, args.trace_steps)[-1] == target

            true_trace = run_program(x, cfg, args.trace_steps)
            mismatch_ok += diagnose_mismatch(packet, x, true_trace)
            corrupted = list(true_trace)
            corrupt_index = corrupt_rng.randrange(1, len(corrupted))
            corrupted[corrupt_index] = (corrupted[corrupt_index] + 1 + corrupt_rng.randrange(cfg.modulus - 1)) % cfg.modulus
            corrupted_accept += diagnose_mismatch(packet, x, corrupted)

            packet_work += (packet_parse_cost / args.heldout_inputs) + args.trace_steps
            control_work += args.train_inputs * args.trace_steps + args.trace_steps

    executable_accuracy = executable_ok / total
    text_only_accuracy = text_ok / total
    trace_nearest_accuracy = nearest_ok / total
    hidden_lookup_accuracy = lookup_ok / total
    shuffled_code_accuracy = shuffled_ok / total
    mismatch_detection_accuracy = mismatch_ok / total
    corrupted_trace_accept_rate = corrupted_accept / total
    packet_work_ratio = packet_work / control_work
    best_control = max(text_only_accuracy, trace_nearest_accuracy, hidden_lookup_accuracy, shuffled_code_accuracy)
    executable_vs_best_control_gain = executable_accuracy / max(best_control, 1e-12)

    passes = (
        executable_accuracy >= args.min_executable_accuracy
        and best_control <= args.max_best_control_accuracy
        and mismatch_detection_accuracy >= args.min_mismatch_detection
        and corrupted_trace_accept_rate <= 1.0 - args.min_mismatch_detection
        and packet_work_ratio <= args.max_packet_work_ratio
    )
    decision = "PASS_EXECUTABLE_CORPUS_PACKETS_GATE" if passes else "FAIL_EXECUTABLE_CORPUS_PACKETS_GATE"
    activation = (
        "[R422_EXECUTABLE_CORPUS_PACKETS_GATE] "
        f"active=1 decision={decision} compiler_material=1 no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        packets=args.packets,
        train_inputs_per_packet=args.train_inputs,
        heldout_inputs_per_packet=args.heldout_inputs,
        trace_steps=args.trace_steps,
        executable_accuracy=executable_accuracy,
        text_only_accuracy=text_only_accuracy,
        trace_nearest_accuracy=trace_nearest_accuracy,
        hidden_lookup_accuracy=hidden_lookup_accuracy,
        shuffled_code_accuracy=shuffled_code_accuracy,
        mismatch_detection_accuracy=mismatch_detection_accuracy,
        corrupted_trace_accept_rate=corrupted_trace_accept_rate,
        packet_work_ratio=packet_work_ratio,
        executable_vs_best_control_gain=executable_vs_best_control_gain,
        activation_log=activation,
        compiler_material=True,
        no_gpu=True,
        no_training_integration=True,
        elapsed_seconds=time.perf_counter() - start,
    )
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
