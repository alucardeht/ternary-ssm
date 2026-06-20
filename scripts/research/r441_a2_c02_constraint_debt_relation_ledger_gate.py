#!/usr/bin/env python3
"""R441 A2-C02 constraint/debt relation ledger gate.

The gate implements the validated A2-C02 object as an observer-only typed
obligation ledger. Source events emit compact relation debts; later demand
events settle or reject those debts against a synthetic exact pair oracle. The
ledger stores relation type, id tag, expiry, and certificate only, not token
payloads. The pass condition follows the validation batch:

constraint_coverage >= 0.80
false_settlement_CE_debt <= 0.10 * saved_CE
ledger payload entropy does not resemble token storage
paid_ratio < 0.90
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


SCHEMA = "r441_a2_c02_constraint_debt_relation_ledger_gate.v1"
RELATION_TYPES = ("scope", "definition_use", "quote_coref", "copy_distractor")


@dataclass(frozen=True)
class Event:
    pos: int
    kind: str
    relation_type: str
    relation_id: str
    payload: str
    is_true_demand: bool = False


@dataclass(frozen=True)
class DebtNote:
    relation_type: str
    relation_id: str
    id_tag: str
    expiry: int
    certificate: str


def digest(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def random_word(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_suite(args: argparse.Namespace) -> list[Event]:
    rng = random.Random(args.seed)
    events: list[Event] = []
    pos = 0
    for idx in range(args.relations):
        relation_type = RELATION_TYPES[idx % len(RELATION_TYPES)]
        rid = f"R{idx:04d}_{random_word(rng, args.id_width)}"
        if relation_type == "scope":
            payload = f"OPEN({rid})"
            demand = f"CLOSE({rid})"
        elif relation_type == "definition_use":
            payload = f"let {rid} = {random_word(rng, args.payload_width)}"
            demand = f"use {rid}"
        elif relation_type == "quote_coref":
            payload = f"quote {rid}: {random_word(rng, args.payload_width)}"
            demand = f"refer {rid}"
        else:
            payload = f"copy {rid}: {random_word(rng, args.payload_width)}"
            demand = f"paste {rid}"
        events.append(Event(pos=pos, kind="emit", relation_type=relation_type, relation_id=rid, payload=payload))
        pos += 1
        if args.mode == "false-control" and idx < args.false_controls:
            wrong_type = RELATION_TYPES[(RELATION_TYPES.index(relation_type) + 1) % len(RELATION_TYPES)]
            events.append(
                Event(
                    pos=pos,
                    kind="demand",
                    relation_type=wrong_type,
                    relation_id=rid,
                    payload=f"false settlement {rid}",
                    is_true_demand=False,
                )
            )
            pos += 1
        for _ in range(args.distractors_per_relation):
            fake_id = f"D{idx:04d}_{random_word(rng, args.id_width)}"
            events.append(
                Event(
                    pos=pos,
                    kind="demand",
                    relation_type=relation_type,
                    relation_id=fake_id,
                    payload=f"distractor {fake_id}",
                    is_true_demand=False,
                )
            )
            pos += 1
        events.append(
            Event(
                pos=pos,
                kind="demand",
                relation_type=relation_type,
                relation_id=rid,
                payload=demand,
                is_true_demand=True,
            )
        )
        pos += 1
    return sorted(events, key=lambda event: event.pos)


def emit_debt(event: Event, args: argparse.Namespace) -> DebtNote:
    expiry = event.pos + args.expiry_radius
    return DebtNote(
        relation_type=event.relation_type,
        relation_id=event.relation_id,
        id_tag=digest(event.relation_id, args.id_tag_hex),
        expiry=expiry,
        certificate=digest(f"{event.relation_type}:{event.relation_id}", args.certificate_hex),
    )


def settle(event: Event, notes: dict[tuple[str, str], DebtNote], args: argparse.Namespace) -> tuple[bool, bool]:
    lookup_type = event.relation_type
    if args.mode == "false-control" and event.kind == "demand" and not event.is_true_demand:
        # Faulty ledger ignores type on false-control demands and may settle the
        # first debt with the same id.
        for (note_type, note_id), note in notes.items():
            if note_id == event.relation_id and event.pos <= note.expiry:
                cert_ok = note.certificate == digest(f"{note_type}:{note_id}", args.certificate_hex)
                return cert_ok, note_type == event.relation_type
    key = (lookup_type, event.relation_id)
    note = notes.get(key)
    if note is None or event.pos > note.expiry:
        return False, False
    cert_ok = note.id_tag == digest(event.relation_id, args.id_tag_hex)
    cert_ok = cert_ok and note.certificate == digest(f"{event.relation_type}:{event.relation_id}", args.certificate_hex)
    return cert_ok, cert_ok


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    events = make_suite(args)
    notes: dict[tuple[str, str], DebtNote] = {}
    emitted = 0
    true_demands = 0
    settled_true = 0
    false_settlements = 0
    rejected_false = 0
    expired_notes = 0
    for event in events:
        expired = [key for key, note in notes.items() if event.pos > note.expiry]
        for key in expired:
            notes.pop(key)
            expired_notes += 1
        if event.kind == "emit":
            notes[(event.relation_type, event.relation_id)] = emit_debt(event, args)
            emitted += 1
            continue
        if event.is_true_demand:
            true_demands += 1
        settled, correct_relation = settle(event, notes, args)
        if settled and event.is_true_demand and correct_relation:
            settled_true += 1
        elif settled and not event.is_true_demand:
            false_settlements += 1
        elif not event.is_true_demand:
            rejected_false += 1

    constraint_coverage = settled_true / max(1, true_demands)
    saved_ce = settled_true * args.saved_ce_per_relation
    false_settlement_ce_debt = false_settlements * args.false_debt_per_settlement
    false_debt_fraction = false_settlement_ce_debt / max(1.0, saved_ce)
    missed_relations = true_demands - settled_true
    missed_repair_cost = missed_relations * args.missed_repair_per_relation
    note_bits = args.type_bits + args.id_tag_bits + args.expiry_bits + args.certificate_bits
    ledger_bits = emitted * note_bits
    verify_bits = len(events) * args.verify_bits_per_event
    raw_relation_payload_bits = emitted * args.raw_payload_tokens * args.bits_per_token
    ledger_payload_entropy_ratio = ledger_bits / max(1.0, raw_relation_payload_bits)
    baseline_relation_ce = true_demands * args.baseline_ce_per_relation
    paid_cost = ledger_bits + verify_bits + missed_repair_cost + false_settlement_ce_debt
    paid_ratio = paid_cost / max(1.0, baseline_relation_ce)
    pass_gate = (
        constraint_coverage >= args.min_constraint_coverage
        and false_debt_fraction <= args.max_false_debt_fraction
        and ledger_payload_entropy_ratio <= args.max_payload_entropy_ratio
        and paid_ratio < args.max_paid_ratio
    )
    decision = (
        "PASS_R477_A2C02_CONSTRAINT_DEBT_RELATION_LEDGER_GATE"
        if pass_gate
        else "FAIL_R477_A2C02_CONSTRAINT_DEBT_RELATION_LEDGER_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "seed": args.seed,
        "relations": args.relations,
        "events": len(events),
        "emitted_debts": emitted,
        "true_demands": true_demands,
        "settled_true": settled_true,
        "constraint_coverage": constraint_coverage,
        "false_settlements": false_settlements,
        "rejected_false_demands": rejected_false,
        "false_settlement_ce_debt": false_settlement_ce_debt,
        "false_debt_fraction": false_debt_fraction,
        "missed_relations": missed_relations,
        "expired_notes": expired_notes,
        "saved_ce": saved_ce,
        "ledger_bits": ledger_bits,
        "verify_bits": verify_bits,
        "missed_repair_cost": missed_repair_cost,
        "paid_cost": paid_cost,
        "baseline_relation_ce": baseline_relation_ce,
        "paid_ratio": paid_ratio,
        "raw_relation_payload_bits": raw_relation_payload_bits,
        "ledger_payload_entropy_ratio": ledger_payload_entropy_ratio,
        "thresholds": {
            "min_constraint_coverage": args.min_constraint_coverage,
            "max_false_debt_fraction": args.max_false_debt_fraction,
            "max_payload_entropy_ratio": args.max_payload_entropy_ratio,
            "max_paid_ratio": args.max_paid_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "observer_only_relation_debt_gate_not_attention_or_kv",
    }


def write_events(path: Path, events: list[Event]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--relations", type=int, default=96)
    parser.add_argument("--distractors-per-relation", type=int, default=2)
    parser.add_argument("--false-controls", type=int, default=16)
    parser.add_argument("--expiry-radius", type=int, default=8)
    parser.add_argument("--seed", type=int, default=44122)
    parser.add_argument("--id-width", type=int, default=6)
    parser.add_argument("--payload-width", type=int, default=8)
    parser.add_argument("--bits-per-token", type=float, default=16.0)
    parser.add_argument("--raw-payload-tokens", type=float, default=10.0)
    parser.add_argument("--type-bits", type=float, default=4.0)
    parser.add_argument("--id-tag-bits", type=float, default=32.0)
    parser.add_argument("--expiry-bits", type=float, default=8.0)
    parser.add_argument("--certificate-bits", type=float, default=32.0)
    parser.add_argument("--verify-bits-per-event", type=float, default=4.0)
    parser.add_argument("--saved-ce-per-relation", type=float, default=96.0)
    parser.add_argument("--baseline-ce-per-relation", type=float, default=192.0)
    parser.add_argument("--missed-repair-per-relation", type=float, default=192.0)
    parser.add_argument("--false-debt-per-settlement", type=float, default=128.0)
    parser.add_argument("--min-constraint-coverage", type=float, default=0.80)
    parser.add_argument("--max-false-debt-fraction", type=float, default=0.10)
    parser.add_argument("--max-payload-entropy-ratio", type=float, default=0.60)
    parser.add_argument("--max-paid-ratio", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-jsonl", type=Path)
    args = parser.parse_args()
    args.id_tag_hex = max(1, int(args.id_tag_bits // 4))
    args.certificate_hex = max(1, int(args.certificate_bits // 4))
    return args


def main() -> int:
    args = parse_args()
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.events_jsonl is not None:
        args.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_events(args.events_jsonl, make_suite(args))
    print(
        "[CONSTRAINT_DEBT_RELATION_LEDGER_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} relations={summary['relations']} events={summary['events']} "
        f"constraint_coverage={summary['constraint_coverage']:.6f} "
        f"false_debt_fraction={summary['false_debt_fraction']:.6f} "
        f"payload_entropy_ratio={summary['ledger_payload_entropy_ratio']:.6f} "
        f"paid_ratio={summary['paid_ratio']:.6f} false_settlements={summary['false_settlements']}"
    )
    print(f"[CONSTRAINT_DEBT_RELATION_LEDGER_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
