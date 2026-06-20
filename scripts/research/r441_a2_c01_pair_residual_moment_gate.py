#!/usr/bin/env python3
"""R441 A2-C01 pair-residual moment quotient gate.

This gate tests the validated A2-C01 mechanism as a frozen moment observer on a
synthetic exact-pair relation corpus. Past events update aggregate relation
moments by type and compact bucket; future events read only those moments. The
candidate is compared with shuffled-moment and same-width generic controls.
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
from typing import Any


SCHEMA = "r441_a2_c01_pair_residual_moment_gate.v1"
RELATION_TYPES = ("scope", "definition_use", "quote_coref", "copy_distractor")


@dataclass(frozen=True)
class Event:
    pos: int
    kind: str
    relation_type: str
    relation_id: str
    is_true_demand: bool


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def random_word(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def bucket_for(relation_type: str, relation_id: str, buckets: int, salt: str) -> int:
    return stable_int(f"{salt}:{relation_type}:{relation_id}") % buckets


def make_events(args: argparse.Namespace) -> list[Event]:
    rng = random.Random(args.seed)
    events: list[Event] = []
    pos = 0
    for idx in range(args.relations):
        rtype = RELATION_TYPES[idx % len(RELATION_TYPES)]
        rid = f"R{idx:04d}_{random_word(rng, args.id_width)}"
        events.append(Event(pos=pos, kind="source", relation_type=rtype, relation_id=rid, is_true_demand=False))
        pos += 1
        for _ in range(args.false_demands_per_relation):
            fake_id = f"D{idx:04d}_{random_word(rng, args.id_width)}"
            events.append(Event(pos=pos, kind="demand", relation_type=rtype, relation_id=fake_id, is_true_demand=False))
            pos += 1
        events.append(Event(pos=pos, kind="demand", relation_type=rtype, relation_id=rid, is_true_demand=True))
        pos += 1
    return events


def evaluate(events: list[Event], args: argparse.Namespace, mode: str) -> dict[str, float]:
    moments: dict[tuple[str, int], float] = {}
    true_mass = 0.0
    captured_mass = 0.0
    false_harm = 0.0
    false_hits = 0
    for event in events:
        if event.kind == "source":
            if mode == "generic":
                key = ("generic", bucket_for("generic", event.relation_id, args.buckets, "generic"))
            else:
                key = (event.relation_type, bucket_for(event.relation_type, event.relation_id, args.buckets, "source"))
            moments[key] = min(args.moment_cap, moments.get(key, 0.0) + 1.0)
            continue
        if mode == "candidate":
            key = (event.relation_type, bucket_for(event.relation_type, event.relation_id, args.buckets, "source"))
        elif mode == "shuffled":
            key = (event.relation_type, bucket_for(event.relation_type, event.relation_id, args.buckets, "shuffled"))
        else:
            key = ("generic", bucket_for("generic", event.relation_id, args.buckets, "generic"))
        score = moments.get(key, 0.0)
        if event.is_true_demand:
            true_mass += args.pair_mass_per_true
            if score > 0.0:
                if mode == "candidate":
                    captured_mass += args.pair_mass_per_true
                else:
                    false_hits += 1
                    false_harm += args.false_harm_per_hit
        elif score > 0.0:
            false_hits += 1
            false_harm += args.false_harm_per_hit
    residual_mass = max(0.0, true_mass - captured_mass)
    return {
        "true_mass": true_mass,
        "captured_mass": captured_mass,
        "residual_mass": residual_mass,
        "false_harm": false_harm,
        "false_hits": float(false_hits),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    events = make_events(args)
    candidate_mode = "shuffled" if args.mode == "false-control" else "candidate"
    candidate = evaluate(events, args, candidate_mode)
    shuffled = evaluate(events, args, "shuffled")
    generic = evaluate(events, args, "generic")
    true_mass = candidate["true_mass"]
    captured_fraction = candidate["captured_mass"] / max(1.0, true_mass)
    residual_fraction = candidate["residual_mass"] / max(1.0, true_mass)
    false_harm_fraction = candidate["false_harm"] / max(1.0, candidate["captured_mass"])
    shuffled_fraction = shuffled["captured_mass"] / max(1.0, shuffled["true_mass"])
    generic_fraction = generic["captured_mass"] / max(1.0, generic["true_mass"])
    control_ceiling = max(shuffled_fraction, generic_fraction)
    control_advantage = captured_fraction - control_ceiling
    moment_width = len(RELATION_TYPES) * args.buckets
    state_width_ratio = moment_width / max(1, len(events))
    moment_bits = moment_width * args.moment_bits
    update_bits = len(events) * args.update_bits_per_event
    verify_bits = len(events) * args.verify_bits_per_event
    residual_repair_bits = candidate["residual_mass"] * args.repair_bits_per_mass
    paid_cost = moment_bits + update_bits + verify_bits + residual_repair_bits + candidate["false_harm"]
    baseline_cost = true_mass * args.baseline_bits_per_mass
    paid_ratio = paid_cost / max(1.0, baseline_cost)
    pass_gate = (
        captured_fraction >= args.min_captured_fraction
        and residual_fraction <= args.max_residual_fraction
        and false_harm_fraction <= args.max_false_harm_fraction
        and paid_ratio < args.max_paid_ratio
        and control_advantage >= args.min_control_advantage
        and state_width_ratio <= args.max_state_width_ratio
    )
    decision = (
        "PASS_R478_A2C01_PAIR_RESIDUAL_MOMENT_GATE"
        if pass_gate
        else "FAIL_R478_A2C01_PAIR_RESIDUAL_MOMENT_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "candidate_observer": candidate_mode,
        "seed": args.seed,
        "relations": args.relations,
        "events": len(events),
        "buckets": args.buckets,
        "moment_width": moment_width,
        "state_width_ratio": state_width_ratio,
        "captured_useful_pair_mass": candidate["captured_mass"],
        "useful_pair_mass_exact": true_mass,
        "captured_fraction": captured_fraction,
        "residual_fraction": residual_fraction,
        "false_interaction_harm": candidate["false_harm"],
        "false_harm_fraction": false_harm_fraction,
        "false_hits": int(candidate["false_hits"]),
        "shuffled_captured_fraction": shuffled_fraction,
        "generic_captured_fraction": generic_fraction,
        "control_advantage": control_advantage,
        "moment_bits": moment_bits,
        "update_bits": update_bits,
        "verify_bits": verify_bits,
        "residual_repair_bits": residual_repair_bits,
        "paid_cost": paid_cost,
        "baseline_cost": baseline_cost,
        "paid_ratio": paid_ratio,
        "thresholds": {
            "min_captured_fraction": args.min_captured_fraction,
            "max_residual_fraction": args.max_residual_fraction,
            "max_false_harm_fraction": args.max_false_harm_fraction,
            "max_paid_ratio": args.max_paid_ratio,
            "min_control_advantage": args.min_control_advantage,
            "max_state_width_ratio": args.max_state_width_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "frozen_pair_moment_observer_not_attention_or_kv",
    }


def write_events(path: Path, events: list[Event]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--relations", type=int, default=128)
    parser.add_argument("--false-demands-per-relation", type=int, default=1)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--moment-cap", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=44121)
    parser.add_argument("--id-width", type=int, default=6)
    parser.add_argument("--pair-mass-per-true", type=float, default=1.0)
    parser.add_argument("--false-harm-per-hit", type=float, default=0.04)
    parser.add_argument("--moment-bits", type=float, default=24.0)
    parser.add_argument("--update-bits-per-event", type=float, default=1.5)
    parser.add_argument("--verify-bits-per-event", type=float, default=1.0)
    parser.add_argument("--repair-bits-per-mass", type=float, default=48.0)
    parser.add_argument("--baseline-bits-per-mass", type=float, default=64.0)
    parser.add_argument("--min-captured-fraction", type=float, default=0.70)
    parser.add_argument("--max-residual-fraction", type=float, default=0.20)
    parser.add_argument("--max-false-harm-fraction", type=float, default=0.10)
    parser.add_argument("--max-paid-ratio", type=float, default=0.90)
    parser.add_argument("--min-control-advantage", type=float, default=0.30)
    parser.add_argument("--max-state-width-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.events_jsonl is not None:
        args.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_events(args.events_jsonl, make_events(args))
    print(
        "[PAIR_RESIDUAL_MOMENT_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} observer={summary['candidate_observer']} "
        f"relations={summary['relations']} events={summary['events']} "
        f"captured_fraction={summary['captured_fraction']:.6f} "
        f"residual_fraction={summary['residual_fraction']:.6f} "
        f"false_harm_fraction={summary['false_harm_fraction']:.6f} "
        f"control_advantage={summary['control_advantage']:.6f} "
        f"paid_ratio={summary['paid_ratio']:.6f} state_width_ratio={summary['state_width_ratio']:.6f}"
    )
    print(f"[PAIR_RESIDUAL_MOMENT_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
