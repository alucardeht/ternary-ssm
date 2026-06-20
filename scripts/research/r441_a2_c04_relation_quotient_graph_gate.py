#!/usr/bin/env python3
"""R441 A2-C04 relation quotient graph / segment-pair quotient gate.

This gate tests the validated A2-C04 object as a relation-native quotient graph
over segment pairs. The candidate groups exact pair effects by typed relation
cells with multiplicity and representative effect; it pays residual replay,
verification, split audit, update, and memory cost. The false-control uses a
broad non-relation-native signature that collapses unlike future demands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


SCHEMA = "r441_a2_c04_relation_quotient_graph_gate.v1"
RELATION_TYPES = ("scope", "definition_use", "quote_coref", "copy_distractor")
DEMAND_CLASSES = ("close", "bind", "refer")
SEGMENT_ROLES = ("header", "body", "tail")


@dataclass(frozen=True)
class SegmentPair:
    pair_id: str
    source_segment: str
    demand_segment: str
    relation_type: str
    demand_class: str
    segment_role: str
    template_id: int
    effect: float
    useful_mass: float


@dataclass
class CellStats:
    key: tuple[str, ...]
    pairs: list[SegmentPair]
    representative: float
    residual_mass: float
    useful_mass: float
    residual_fraction: float
    split_gain: float


def digest(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def stable_unit(text: str) -> float:
    raw = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return raw / float(16**12 - 1)


def random_word(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_pairs(args: argparse.Namespace) -> list[SegmentPair]:
    rng = random.Random(args.seed)
    pairs: list[SegmentPair] = []
    base_by_type = {
        "scope": 0.92,
        "definition_use": 1.14,
        "quote_coref": 1.36,
        "copy_distractor": 1.58,
    }
    demand_delta = {"close": -0.08, "bind": 0.0, "refer": 0.08}
    role_delta = {"header": -0.025, "body": 0.0, "tail": 0.025}
    for idx in range(args.pairs):
        relation_type = RELATION_TYPES[idx % len(RELATION_TYPES)]
        demand_class = DEMAND_CLASSES[(idx // len(RELATION_TYPES)) % len(DEMAND_CLASSES)]
        segment_role = SEGMENT_ROLES[(idx // (len(RELATION_TYPES) * len(DEMAND_CLASSES))) % len(SEGMENT_ROLES)]
        template_id = idx % args.templates
        relation_id = f"R{idx:04d}_{random_word(rng, args.id_width)}"
        source_segment = f"{segment_role}:{relation_type}:source:{relation_id}:t{template_id}"
        demand_segment = f"{demand_class}:{relation_type}:demand:{relation_id}:t{template_id}"
        jitter = (stable_unit(f"{args.seed}:{relation_id}") - 0.5) * args.jitter_width
        effect = base_by_type[relation_type] + demand_delta[demand_class] + role_delta[segment_role] + jitter
        useful_mass = args.mass_scale * (1.0 + 0.05 * (template_id % 3))
        pairs.append(
            SegmentPair(
                pair_id=digest(f"{source_segment}->{demand_segment}"),
                source_segment=source_segment,
                demand_segment=demand_segment,
                relation_type=relation_type,
                demand_class=demand_class,
                segment_role=segment_role,
                template_id=template_id,
                effect=effect,
                useful_mass=useful_mass,
            )
        )
    return pairs


def relation_native_key(pair: SegmentPair, args: argparse.Namespace) -> tuple[str, ...]:
    template_bucket = str(pair.template_id % args.template_buckets)
    return ("relation", pair.relation_type, pair.demand_class, pair.segment_role, template_bucket)


def broad_state_hash_key(pair: SegmentPair, args: argparse.Namespace) -> tuple[str, ...]:
    # Deliberately omits relation_type and demand_class. This mimics the failed
    # broad state-hash quotient family and should be caught by split audits.
    bucket = str(pair.template_id % args.generic_buckets)
    return ("generic", pair.segment_role, bucket)


def group_pairs(pairs: list[SegmentPair], args: argparse.Namespace, mode: str) -> dict[tuple[str, ...], list[SegmentPair]]:
    groups: dict[tuple[str, ...], list[SegmentPair]] = {}
    for pair in pairs:
        key = broad_state_hash_key(pair, args) if mode == "broad" else relation_native_key(pair, args)
        groups.setdefault(key, []).append(pair)
    return groups


def weighted_mean(pairs: list[SegmentPair]) -> float:
    total_mass = sum(pair.useful_mass for pair in pairs)
    return sum(pair.effect * pair.useful_mass for pair in pairs) / max(1e-12, total_mass)


def cell_residual(pairs: list[SegmentPair], representative: float) -> float:
    return sum(abs(pair.effect - representative) * pair.useful_mass for pair in pairs)


def split_residual(pairs: list[SegmentPair]) -> float:
    split_groups: dict[tuple[str, str], list[SegmentPair]] = {}
    for pair in pairs:
        split_groups.setdefault((pair.relation_type, pair.demand_class), []).append(pair)
    return sum(cell_residual(split_pairs, weighted_mean(split_pairs)) for split_pairs in split_groups.values())


def evaluate(pairs: list[SegmentPair], args: argparse.Namespace, mode: str) -> dict[str, Any]:
    groups = group_pairs(pairs, args, mode)
    cells: list[CellStats] = []
    total_useful_mass = sum(pair.useful_mass for pair in pairs)
    for key, cell_pairs in groups.items():
        representative = weighted_mean(cell_pairs)
        residual = cell_residual(cell_pairs, representative)
        useful = sum(pair.useful_mass for pair in cell_pairs)
        split_after = split_residual(cell_pairs)
        cells.append(
            CellStats(
                key=key,
                pairs=cell_pairs,
                representative=representative,
                residual_mass=residual,
                useful_mass=useful,
                residual_fraction=residual / max(1e-12, useful),
                split_gain=max(0.0, residual - split_after),
            )
        )
    high_mass_cells = [cell for cell in cells if cell.useful_mass / max(1.0, total_useful_mass) >= args.high_mass_cell_floor]
    low_residual_cells = [cell for cell in cells if cell.residual_fraction <= args.max_cell_residual_fraction]
    accepted_mass = sum(cell.useful_mass for cell in low_residual_cells)
    total_residual = sum(cell.residual_mass for cell in cells)
    total_split_gain = sum(cell.split_gain for cell in cells)
    high_mass_cell_residual = max((cell.residual_fraction for cell in high_mass_cells), default=1.0)
    accepted_mass_fraction = accepted_mass / max(1.0, total_useful_mass)
    residual_fraction = total_residual / max(1.0, total_useful_mass)
    split_gain_fraction = total_split_gain / max(1.0, total_useful_mass)
    false_equivalence_debt = max(0.0, split_gain_fraction - args.max_split_gain_fraction) * total_useful_mass
    quotient_cells = len(cells)
    quotient_edges = sum(len(set((pair.relation_type, pair.demand_class) for pair in cell.pairs)) for cell in cells)
    memory_bits = quotient_cells * args.cell_bits
    edge_bits = quotient_edges * args.edge_bits
    update_bits = len(pairs) * args.update_bits_per_pair
    verify_bits = len(pairs) * args.verify_bits_per_pair
    split_audit_bits = len(cells) * args.split_audit_bits_per_cell
    residual_replay_bits = total_residual * args.residual_replay_bits_per_mass
    false_debt_bits = false_equivalence_debt * args.false_equivalence_bits_per_mass
    paid_cost = memory_bits + edge_bits + update_bits + verify_bits + split_audit_bits + residual_replay_bits + false_debt_bits
    baseline_cost = total_useful_mass * args.baseline_bits_per_mass
    paid_ratio = paid_cost / max(1.0, baseline_cost)
    state_width_ratio = quotient_cells / max(1, len(pairs))
    pass_gate = (
        mode == "relation-native"
        and high_mass_cell_residual <= args.max_cell_residual_fraction
        and accepted_mass_fraction >= args.min_accepted_mass_fraction
        and residual_fraction <= args.max_total_residual_fraction
        and split_gain_fraction <= args.max_split_gain_fraction
        and paid_ratio < args.max_paid_ratio
        and state_width_ratio <= args.max_state_width_ratio
    )
    return {
        "mode": mode,
        "relation_native_signature": mode == "relation-native",
        "quotient_cells": quotient_cells,
        "quotient_edges": quotient_edges,
        "total_useful_pair_mass": total_useful_mass,
        "accepted_useful_pair_mass": accepted_mass,
        "accepted_mass_fraction": accepted_mass_fraction,
        "total_residual_mass": total_residual,
        "residual_fraction": residual_fraction,
        "high_mass_cell_residual": high_mass_cell_residual,
        "split_gain_fraction": split_gain_fraction,
        "false_equivalence_debt": false_equivalence_debt,
        "memory_bits": memory_bits,
        "edge_bits": edge_bits,
        "update_bits": update_bits,
        "verify_bits": verify_bits,
        "split_audit_bits": split_audit_bits,
        "residual_replay_bits": residual_replay_bits,
        "false_debt_bits": false_debt_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "state_width_ratio": state_width_ratio,
        "cell_size_mean": fmean([len(cell.pairs) for cell in cells]) if cells else 0.0,
        "cell_size_max": max((len(cell.pairs) for cell in cells), default=0),
        "cell_residual_mean": fmean([cell.residual_fraction for cell in cells]) if cells else 1.0,
        "pass": pass_gate,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    pairs = make_pairs(args)
    candidate_mode = "broad" if args.mode == "false-control" else "relation-native"
    candidate = evaluate(pairs, args, candidate_mode)
    broad = evaluate(pairs, args, "broad")
    relation_native = evaluate(pairs, args, "relation-native")
    pass_gate = bool(candidate["pass"])
    decision = (
        "PASS_R479_A2C04_RELATION_QUOTIENT_GRAPH_GATE"
        if pass_gate
        else "FAIL_R479_A2C04_RELATION_QUOTIENT_GRAPH_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "candidate_signature": candidate_mode,
        "seed": args.seed,
        "pairs": args.pairs,
        "templates": args.templates,
        "template_buckets": args.template_buckets,
        "generic_buckets": args.generic_buckets,
        "candidate": candidate,
        "relation_native_control": relation_native,
        "broad_state_hash_control": broad,
        "thresholds": {
            "max_cell_residual_fraction": args.max_cell_residual_fraction,
            "min_accepted_mass_fraction": args.min_accepted_mass_fraction,
            "max_total_residual_fraction": args.max_total_residual_fraction,
            "max_split_gain_fraction": args.max_split_gain_fraction,
            "max_paid_ratio": args.max_paid_ratio,
            "max_state_width_ratio": args.max_state_width_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "relation_native_segment_pair_quotient_graph_not_attention_or_state_hash",
    }


def write_pairs(path: Path, pairs: list[SegmentPair]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--pairs", type=int, default=216)
    parser.add_argument("--templates", type=int, default=18)
    parser.add_argument("--template-buckets", type=int, default=3)
    parser.add_argument("--generic-buckets", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44124)
    parser.add_argument("--id-width", type=int, default=6)
    parser.add_argument("--jitter-width", type=float, default=0.025)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--high-mass-cell-floor", type=float, default=0.01)
    parser.add_argument("--cell-bits", type=float, default=18.0)
    parser.add_argument("--edge-bits", type=float, default=4.0)
    parser.add_argument("--update-bits-per-pair", type=float, default=1.0)
    parser.add_argument("--verify-bits-per-pair", type=float, default=1.0)
    parser.add_argument("--split-audit-bits-per-cell", type=float, default=2.0)
    parser.add_argument("--residual-replay-bits-per-mass", type=float, default=64.0)
    parser.add_argument("--false-equivalence-bits-per-mass", type=float, default=96.0)
    parser.add_argument("--baseline-bits-per-mass", type=float, default=64.0)
    parser.add_argument("--max-cell-residual-fraction", type=float, default=0.20)
    parser.add_argument("--min-accepted-mass-fraction", type=float, default=0.30)
    parser.add_argument("--max-total-residual-fraction", type=float, default=0.20)
    parser.add_argument("--max-split-gain-fraction", type=float, default=0.10)
    parser.add_argument("--max-paid-ratio", type=float, default=0.90)
    parser.add_argument("--max-state-width-ratio", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.pairs_jsonl is not None:
        args.pairs_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_pairs(args.pairs_jsonl, make_pairs(args))
    candidate = result["candidate"]
    print(
        "[RELATION_QUOTIENT_GRAPH_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass'])} "
        f"mode={result['mode']} signature={result['candidate_signature']} "
        f"pairs={result['pairs']} cells={candidate['quotient_cells']} "
        f"accepted_mass_fraction={candidate['accepted_mass_fraction']:.6f} "
        f"high_mass_cell_residual={candidate['high_mass_cell_residual']:.6f} "
        f"residual_fraction={candidate['residual_fraction']:.6f} "
        f"split_gain_fraction={candidate['split_gain_fraction']:.6f} "
        f"paid_ratio={candidate['paid_ratio']:.6f} "
        f"state_width_ratio={candidate['state_width_ratio']:.6f}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
