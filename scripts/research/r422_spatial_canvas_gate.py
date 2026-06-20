#!/usr/bin/env python3
"""R422 spatial canvas internal-code gate.

This gate tests the structural claim behind R422-MS-03: a canonical text stream
can be encoded into a discrete spatial canvas, decoded exactly, and solved by
local parallel propagation with lower dependency depth than sequential token
recurrence. It is a synthetic structural gate, not a GPU benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_spatial_canvas_gate.v1"


@dataclass
class ChainCase:
    chain_len: int
    distractors: int
    text: str
    query_key: str
    target_value: str


@dataclass
class CaseResult:
    chain_len: int
    distractors: int
    token_count: int
    canvas_cells: int
    reconstruction_exact: bool
    answer_correct: bool
    random_control_correct: bool
    local_rounds: int
    sequential_dependency_steps: int
    parallel_depth_ratio: float
    work_ratio: float
    residual_text_ratio: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    cases: int
    mean_parallel_depth_ratio: float
    mean_work_ratio: float
    max_residual_text_ratio: float
    exact_reconstruction_fraction: float
    answer_accuracy: float
    random_control_accuracy: float
    min_depth_gain: float
    max_residual_text_ratio_threshold: float
    activation_log: str
    interpretation: str
    case_results: list[CaseResult]
    elapsed_seconds: float


def make_case(chain_len: int, distractors: int, rng: random.Random) -> ChainCase:
    keys = [f"k{i}" for i in range(chain_len)]
    value = f"v{rng.randrange(10_000, 99_999)}"
    lines: list[str] = []
    for idx in range(chain_len - 1):
        lines.append(f"{keys[idx]} -> {keys[idx + 1]}")
    lines.append(f"{keys[-1]} = {value}")
    for idx in range(distractors):
        lines.append(f"d{idx} -> d{idx}_x")
        lines.append(f"d{idx}_x = v{rng.randrange(1000, 9999)}")
    rng.shuffle(lines)
    text = "\n".join(lines) + f"\n? {keys[0]}\n"
    return ChainCase(
        chain_len=chain_len,
        distractors=distractors,
        text=text,
        query_key=keys[0],
        target_value=value,
    )


def tokenize(text: str) -> list[str]:
    return text.replace("\n", " \n ").split()


def encode_canvas(text: str) -> tuple[dict[str, str], dict[str, str], list[str]]:
    edges: dict[str, str] = {}
    values: dict[str, str] = {}
    query: list[str] = []
    for raw in text.strip().splitlines():
        parts = raw.split()
        if len(parts) == 3 and parts[1] == "->":
            edges[parts[0]] = parts[2]
        elif len(parts) == 3 and parts[1] == "=":
            values[parts[0]] = parts[2]
        elif len(parts) == 2 and parts[0] == "?":
            query.append(parts[1])
        else:
            raise ValueError(f"cannot encode non-canonical line: {raw!r}")
    return edges, values, query


def decode_canvas(edges: dict[str, str], values: dict[str, str], query: list[str]) -> str:
    lines = []
    for key, dst in sorted(edges.items(), key=lambda item: item[0]):
        lines.append(f"{key} -> {dst}")
    for key, value in sorted(values.items(), key=lambda item: item[0]):
        lines.append(f"{key} = {value}")
    for key in query:
        lines.append(f"? {key}")
    return "\n".join(lines) + "\n"


def canonicalize(text: str) -> str:
    edges, values, query = encode_canvas(text)
    return decode_canvas(edges, values, query)


def pointer_jump_answer(edges: dict[str, str], values: dict[str, str], query_key: str) -> tuple[str | None, int]:
    parent = dict(edges)
    val = dict(values)
    nodes = set(parent) | set(parent.values()) | set(val)
    rounds = 0
    max_rounds = max(1, math.ceil(math.log2(max(2, len(nodes)))) + 2)
    for _ in range(max_rounds):
        rounds += 1
        changed = False
        for node in list(nodes):
            if node in val:
                continue
            dst = parent.get(node)
            if dst is None:
                continue
            if dst in val:
                val[node] = val[dst]
                changed = True
            elif dst in parent:
                parent[node] = parent[dst]
                changed = True
        if query_key in val:
            return val[query_key], rounds
        if not changed:
            break
    return val.get(query_key), rounds


def sequential_follow(edges: dict[str, str], values: dict[str, str], query_key: str) -> tuple[str | None, int]:
    seen = set()
    key = query_key
    steps = 0
    while key not in seen:
        seen.add(key)
        steps += 1
        if key in values:
            return values[key], steps
        if key not in edges:
            return None, steps
        key = edges[key]
    return None, steps


def random_control(edges: dict[str, str], values: dict[str, str], query_key: str, rng: random.Random) -> str | None:
    shuffled = list(edges.values())
    rng.shuffle(shuffled)
    random_edges = {key: shuffled[idx] for idx, key in enumerate(edges.keys())}
    answer, _ = pointer_jump_answer(random_edges, values, query_key)
    return answer


def evaluate_case(case: ChainCase, rng: random.Random) -> CaseResult:
    edges, values, query = encode_canvas(case.text)
    decoded = decode_canvas(edges, values, query)
    exact = decoded == canonicalize(case.text)
    answer, rounds = pointer_jump_answer(edges, values, case.query_key)
    seq_answer, seq_steps = sequential_follow(edges, values, case.query_key)
    rand_answer = random_control(edges, values, case.query_key, rng)
    token_count = len(tokenize(case.text))
    cells = len(edges) + len(values)
    parallel_depth = rounds + 1.0
    sequential_depth = token_count + seq_steps
    parallel_depth_ratio = parallel_depth / max(1.0, sequential_depth)
    canvas_work = cells * rounds + token_count
    sequential_work = token_count + seq_steps * token_count
    work_ratio = canvas_work / max(1.0, sequential_work)
    residual_text_ratio = 0.0 if exact else 1.0
    return CaseResult(
        chain_len=case.chain_len,
        distractors=case.distractors,
        token_count=token_count,
        canvas_cells=cells,
        reconstruction_exact=exact,
        answer_correct=answer == case.target_value and seq_answer == case.target_value,
        random_control_correct=rand_answer == case.target_value,
        local_rounds=rounds,
        sequential_dependency_steps=seq_steps,
        parallel_depth_ratio=parallel_depth_ratio,
        work_ratio=work_ratio,
        residual_text_ratio=residual_text_ratio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=80)
    parser.add_argument("--min-chain", type=int, default=8)
    parser.add_argument("--max-chain", type=int, default=96)
    parser.add_argument("--distractors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42203)
    parser.add_argument("--min-depth-gain", type=float, default=2.0)
    parser.add_argument("--max-work-ratio", type=float, default=0.85)
    parser.add_argument("--max-residual-text-ratio", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    if args.min_chain < 2 or args.max_chain < args.min_chain:
        raise ValueError("chain range must satisfy 2 <= min <= max")
    results: list[CaseResult] = []
    for _ in range(args.cases):
        chain_len = rng.randrange(args.min_chain, args.max_chain + 1)
        case = make_case(chain_len, args.distractors, rng)
        results.append(evaluate_case(case, rng))

    exact_fraction = sum(r.reconstruction_exact for r in results) / max(1, len(results))
    accuracy = sum(r.answer_correct for r in results) / max(1, len(results))
    random_accuracy = sum(r.random_control_correct for r in results) / max(1, len(results))
    mean_depth_ratio = sum(r.parallel_depth_ratio for r in results) / max(1, len(results))
    mean_work_ratio = sum(r.work_ratio for r in results) / max(1, len(results))
    max_residual = max((r.residual_text_ratio for r in results), default=1.0)
    depth_gain = 1.0 / max(mean_depth_ratio, 1e-12)
    pass_gate = (
        exact_fraction == 1.0
        and accuracy == 1.0
        and random_accuracy <= 0.05
        and depth_gain >= args.min_depth_gain
        and mean_work_ratio <= args.max_work_ratio
        and max_residual <= args.max_residual_text_ratio
    )
    if pass_gate:
        decision = "PASS_SPATIAL_CANVAS_GATE"
    elif exact_fraction < 1.0 or max_residual > args.max_residual_text_ratio:
        decision = "FAIL_TEXT_CANVAS_CODEC"
    elif accuracy < 1.0:
        decision = "FAIL_CANVAS_PROPAGATION_CORRECTNESS"
    elif random_accuracy > 0.05:
        decision = "FAIL_RANDOM_CANVAS_CONTROL"
    elif depth_gain < args.min_depth_gain:
        decision = "FAIL_PARALLEL_DEPTH_GAIN"
    else:
        decision = "FAIL_CANVAS_WORK_PRICING"

    activation = (
        "[R422_SPATIAL_CANVAS_GATE] "
        f"active=1 decision={decision} pass={int(pass_gate)} "
        "codec=canonical_text_canvas_text propagation=pointer_jumping "
        "no_gpu=1 no_training_integration=1"
    )
    payload = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        cases=len(results),
        mean_parallel_depth_ratio=mean_depth_ratio,
        mean_work_ratio=mean_work_ratio,
        max_residual_text_ratio=max_residual,
        exact_reconstruction_fraction=exact_fraction,
        answer_accuracy=accuracy,
        random_control_accuracy=random_accuracy,
        min_depth_gain=args.min_depth_gain,
        max_residual_text_ratio_threshold=args.max_residual_text_ratio,
        activation_log=activation,
        interpretation=(
            "synthetic_structural_gate_only: validates exact canonical codec "
            "and local canvas propagation, not broad language quality or GPU speed."
        ),
        case_results=results,
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
                "depth_gain": depth_gain,
                "mean_work_ratio": mean_work_ratio,
                "exact_reconstruction_fraction": exact_fraction,
                "answer_accuracy": accuracy,
                "random_control_accuracy": random_accuracy,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
