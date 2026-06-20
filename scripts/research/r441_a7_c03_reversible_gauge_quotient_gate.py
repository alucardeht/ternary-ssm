#!/usr/bin/env python3
"""R441 A7-C03 reversible local-gauge quotient gate.

This gate tests the dataset-object claim before any training run: structured
source spans may be converted into a canonical local-gauge representative plus
a paid sidecar table while preserving the original transcript exactly.

The admissible object is the pair (C, S) with deterministic realization
D(C, S) = X. The reported loss bound is therefore a proxy accounting bound:
the original target can be charged to the canonical stream plus sidecar stream
when the inverse audit is exact. No external labels or teacher states are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "r441_a7_c03_reversible_gauge_quotient_gate.v1"
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")
KEY_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]{5,})"\s*:')
RESERVED = {
    "return",
    "import",
    "from",
    "while",
    "class",
    "def",
    "for",
    "if",
    "else",
    "elif",
    "true",
    "false",
    "null",
}


@dataclass(frozen=True)
class SourceSpan:
    span_id: int
    family: str
    text: str


@dataclass
class GaugeObject:
    span_id: int
    family: str
    transformed: bool
    original_hash: str
    canonical: str
    sidecar: list[tuple[str, str]]
    original_cost: float
    canonical_cost: float
    sidecar_cost: float
    audit_cost: float
    roundtrip_ok: bool
    reconstructed_hash: str
    covered_original_cost: float

    @property
    def object_cost(self) -> float:
        return self.canonical_cost + self.sidecar_cost + self.audit_cost


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def token_proxy_cost(text: str) -> float:
    """Cheap BPE-like proxy that charges long identifiers more than placeholders."""

    cost = 0.0
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\s+|.", text, re.DOTALL):
        token = match.group(0)
        if token.isspace():
            cost += 1.0
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            cost += max(1.0, math.ceil(len(token) / 6.0))
        elif token.isdigit():
            cost += max(1.0, math.ceil(len(token) / 4.0))
        else:
            cost += 1.0
    return cost


def sidecar_proxy_cost(sidecar: list[tuple[str, str]]) -> float:
    if not sidecar:
        return 0.0
    return sum(1.0 + token_proxy_cost(value) for _, value in sidecar)


def make_code_span(span_id: int, rng: random.Random) -> SourceSpan:
    nouns = [
        "inventory_reconciliation_window",
        "customer_balance_delta",
        "warehouse_transfer_receipt",
        "projected_liability_bucket",
        "session_authorization_token",
        "regional_capacity_forecast",
    ]
    primary = rng.choice(nouns) + f"_{span_id}"
    secondary = rng.choice([n for n in nouns if not primary.startswith(n)]) + f"_{span_id}"
    text = (
        f"def update_{primary}(record, ledger):\n"
        f"    {primary} = record.get('{primary}', 0)\n"
        f"    {secondary} = ledger.get('{secondary}', 0)\n"
        f"    if {primary} > {secondary}:\n"
        f"        ledger['{secondary}'] = {primary} - {secondary}\n"
        f"    return {primary}, {secondary}, ledger['{secondary}']\n"
    )
    return SourceSpan(span_id=span_id, family="code_local_identifiers", text=text)


def make_json_span(span_id: int, rng: random.Random) -> SourceSpan:
    keys = [
        "customer_balance_delta",
        "shipment_authorization_state",
        "inventory_reconciliation_window",
        "projected_liability_bucket",
    ]
    rng.shuffle(keys)
    rows = []
    for row in range(4):
        fields = []
        for key in keys[:3]:
            value = (span_id * 17 + row * 5 + len(key)) % 97
            fields.append(f'    "{key}": {value}')
        rows.append("{\n" + ",\n".join(fields) + "\n  }")
    text = "{\n  \"records\": [\n  " + ",\n  ".join(rows) + "\n  ]\n}\n"
    return SourceSpan(span_id=span_id, family="json_repeated_keys", text=text)


def make_table_span(span_id: int, rng: random.Random) -> SourceSpan:
    columns = [
        "regional_capacity_forecast",
        "warehouse_transfer_receipt",
        "customer_balance_delta",
    ]
    rng.shuffle(columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in range(5):
        values = [str((span_id + row + idx * 11) % 41) for idx in range(len(columns))]
        rows.append("| " + " | ".join(values) + " |")
    return SourceSpan(span_id=span_id, family="markdown_table_headers", text="\n".join([header, rule, *rows]) + "\n")


def make_raw_span(span_id: int, rng: random.Random) -> SourceSpan:
    topics = [
        "The small probe keeps unstructured narrative text as a raw fallback.",
        "Fallback spans must still pay the original transcript cost exactly.",
        "This paragraph intentionally has no repeated local gauge labels.",
    ]
    rng.shuffle(topics)
    return SourceSpan(span_id=span_id, family="raw_fallback", text=" ".join(topics) + "\n")


def make_corpus(spans: int, seed: int) -> list[SourceSpan]:
    rng = random.Random(seed)
    corpus: list[SourceSpan] = []
    for span_id in range(spans):
        family_choice = span_id % 8
        if family_choice in {0, 1, 2}:
            corpus.append(make_code_span(span_id, rng))
        elif family_choice in {3, 4, 5}:
            corpus.append(make_json_span(span_id, rng))
        elif family_choice == 6:
            corpus.append(make_table_span(span_id, rng))
        else:
            corpus.append(make_raw_span(span_id, rng))
    return corpus


def choose_gauge_terms(text: str, min_occurrences: int, min_identifier_len: int) -> list[str]:
    terms: dict[str, int] = {}
    for match in IDENT_RE.finditer(text):
        term = match.group(0)
        if len(term) < min_identifier_len or term.lower() in RESERVED:
            continue
        terms[term] = terms.get(term, 0) + 1
    for match in KEY_RE.finditer(text):
        term = match.group(1)
        if len(term) >= min_identifier_len:
            terms[term] = terms.get(term, 0) + 1
    candidates = [term for term, count in terms.items() if count >= min_occurrences]
    candidates.sort(key=lambda item: (-terms[item], item))
    return candidates


def realize(canonical: str, sidecar: list[tuple[str, str]]) -> str:
    restored = canonical
    for placeholder, original in reversed(sidecar):
        restored = restored.replace(placeholder, original)
    return restored


def canonicalize(span: SourceSpan, min_occurrences: int, min_identifier_len: int, audit_cost: float) -> GaugeObject:
    terms = choose_gauge_terms(span.text, min_occurrences, min_identifier_len)
    canonical = span.text
    sidecar: list[tuple[str, str]] = []
    for idx, term in enumerate(terms):
        placeholder = f"__R471G{idx}__"
        if placeholder in canonical:
            continue
        canonical = re.sub(rf"\b{re.escape(term)}\b", placeholder, canonical)
        sidecar.append((placeholder, term))

    original_cost = token_proxy_cost(span.text)
    canonical_cost = token_proxy_cost(canonical)
    sc_cost = sidecar_proxy_cost(sidecar)
    reconstructed = realize(canonical, sidecar)
    transformed = bool(sidecar) and (canonical_cost + sc_cost + audit_cost < original_cost)
    if not transformed:
        canonical = span.text
        sidecar = []
        canonical_cost = original_cost
        sc_cost = 0.0
        reconstructed = span.text
    return GaugeObject(
        span_id=span.span_id,
        family=span.family,
        transformed=transformed,
        original_hash=stable_hash(span.text),
        canonical=canonical,
        sidecar=sidecar,
        original_cost=original_cost,
        canonical_cost=canonical_cost,
        sidecar_cost=sc_cost,
        audit_cost=audit_cost if transformed else 0.0,
        roundtrip_ok=reconstructed == span.text,
        reconstructed_hash=stable_hash(reconstructed),
        covered_original_cost=original_cost if transformed else 0.0,
    )


def corrupt_first_sidecar(objects: list[GaugeObject]) -> None:
    for obj in objects:
        if obj.sidecar:
            placeholder, original = obj.sidecar[0]
            obj.sidecar[0] = (placeholder, original + "_corrupt")
            reconstructed = realize(obj.canonical, obj.sidecar)
            obj.roundtrip_ok = reconstructed == ""
            obj.reconstructed_hash = stable_hash(reconstructed)
            return


def summarize(objects: list[GaugeObject], args: argparse.Namespace, elapsed_s: float) -> dict[str, Any]:
    total_original = sum(obj.original_cost for obj in objects)
    total_canonical = sum(obj.canonical_cost for obj in objects)
    total_sidecar = sum(obj.sidecar_cost for obj in objects)
    total_audit = sum(obj.audit_cost for obj in objects)
    covered_original = sum(obj.covered_original_cost for obj in objects)
    transformed_count = sum(1 for obj in objects if obj.transformed)
    roundtrip_failures = [obj.span_id for obj in objects if not obj.roundtrip_ok]
    total_paid = total_canonical + total_sidecar + total_audit
    coverage = covered_original / total_original if total_original > 0 else 0.0
    sidecar_density = total_sidecar / covered_original if covered_original > 0 else 1.0
    canonical_density = total_canonical / covered_original if covered_original > 0 else 1.0
    loss_bound_ratio = total_paid / total_original if total_original > 0 else float("inf")
    priced_gain = total_original / total_paid if total_paid > 0 else 0.0
    bpe_expansion = total_paid / total_original if total_original > 0 else float("inf")
    exact_roundtrip = not roundtrip_failures
    pass_gate = (
        exact_roundtrip
        and coverage >= args.min_coverage
        and sidecar_density <= args.max_sidecar_density
        and priced_gain >= args.min_priced_gain
        and bpe_expansion <= args.max_bpe_expansion
    )
    decision = (
        "PASS_R471_A7C03_REVERSIBLE_GAUGE_QUOTIENT_GATE"
        if pass_gate
        else "FAIL_R471_A7C03_REVERSIBLE_GAUGE_QUOTIENT_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "seed": args.seed,
        "spans": len(objects),
        "transformed_spans": transformed_count,
        "roundtrip_failures": len(roundtrip_failures),
        "roundtrip_failure_ids": roundtrip_failures[:16],
        "exact_roundtrip": exact_roundtrip,
        "total_original_proxy_loss": total_original,
        "total_canonical_proxy_loss": total_canonical,
        "total_sidecar_proxy_loss": total_sidecar,
        "total_audit_proxy_loss": total_audit,
        "total_paid_proxy_loss": total_paid,
        "covered_original_proxy_loss": covered_original,
        "coverage": coverage,
        "canonical_density": canonical_density,
        "sidecar_density": sidecar_density,
        "loss_bound_ratio": loss_bound_ratio,
        "priced_gain": priced_gain,
        "bpe_expansion": bpe_expansion,
        "thresholds": {
            "min_coverage": args.min_coverage,
            "max_sidecar_density": args.max_sidecar_density,
            "min_priced_gain": args.min_priced_gain,
            "max_bpe_expansion": args.max_bpe_expansion,
        },
        "elapsed_s": elapsed_s,
        "claim_scope": "reversible_dataset_object_gate_not_training_speed",
    }


def write_jsonl(path: Path, objects: list[GaugeObject]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for obj in objects:
            row = asdict(obj)
            row["canonical_preview"] = row.pop("canonical")[:240]
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--spans", type=int, default=64)
    parser.add_argument("--seed", type=int, default=44173)
    parser.add_argument("--min-occurrences", type=int, default=3)
    parser.add_argument("--min-identifier-len", type=int, default=12)
    parser.add_argument("--audit-cost", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.15)
    parser.add_argument("--max-sidecar-density", type=float, default=0.20)
    parser.add_argument("--min-priced-gain", type=float, default=1.10)
    parser.add_argument("--max-bpe-expansion", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    corpus = make_corpus(args.spans, args.seed)
    objects = [
        canonicalize(
            span,
            min_occurrences=args.min_occurrences,
            min_identifier_len=args.min_identifier_len,
            audit_cost=args.audit_cost,
        )
        for span in corpus
    ]
    if args.mode == "false-control":
        corrupt_first_sidecar(objects)
    elapsed = time.perf_counter() - start
    summary = summarize(objects, args, elapsed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.objects_jsonl is not None:
        args.objects_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.objects_jsonl, objects)

    print(
        "[REVERSIBLE_GAUGE_QUOTIENT_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} spans={summary['spans']} "
        f"transformed_spans={summary['transformed_spans']} "
        f"exact_roundtrip={int(summary['exact_roundtrip'])} "
        f"roundtrip_failures={summary['roundtrip_failures']} "
        f"coverage={summary['coverage']:.6f} "
        f"sidecar_density={summary['sidecar_density']:.6f} "
        f"loss_bound_ratio={summary['loss_bound_ratio']:.6f} "
        f"priced_gain={summary['priced_gain']:.6f} "
        f"bpe_expansion={summary['bpe_expansion']:.6f}"
    )
    print(f"[REVERSIBLE_GAUGE_QUOTIENT_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
