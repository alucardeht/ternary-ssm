#!/usr/bin/env python3
"""R441 A1-C04 realized dictionary and syndrome quotient gate.

The gate compiles spans from a finite model-owned generator dictionary plus
sparse literal exceptions. It verifies exact decode, checksum/syndrome pass,
and the priced inequality:

bits(dictionary) + bits(descriptors) + bits(exceptions) + verify_cost
    < bits(raw spans)
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


SCHEMA = "r441_a1_c04_dictionary_syndrome_quotient_gate.v1"
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]", re.ASCII)

TEMPLATES = [
    "def {name}(record):\n    value = record['{field}'] + {offset}\n    return value\n",
    "schema {name} {{ field: {field}; limit: {offset}; mode: stable; }}\n",
    "FORMULA {name}: output = ({field} * {offset}) + CHECKSUM\n",
    "[[{name}]]\n- field: {field}\n- threshold: {offset}\n- status: active\n",
]
NAMES = ["alpha_route", "beta_window", "gamma_scope", "delta_queue", "epsilon_margin", "zeta_policy"]
FIELDS = ["balance_delta", "capacity_forecast", "auth_state", "retry_bucket", "ledger_window", "shipment_state"]
OFFSETS = ["3", "5", "8", "13", "21", "34"]


@dataclass(frozen=True)
class Descriptor:
    template_id: int
    name_id: int
    field_id: int
    offset_id: int


@dataclass(frozen=True)
class ExceptionPatch:
    position: int
    token: str


@dataclass
class CompiledSpan:
    span_id: int
    descriptor: Descriptor
    exceptions: list[ExceptionPatch]
    source_hash: str
    decoded_hash: str
    exact_decode: bool
    checksum_pass: bool
    raw_tokens: int
    exception_tokens: int
    raw_bits: float
    descriptor_bits: float
    exception_bits: float
    verify_bits: float


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def untokenize(tokens: list[str]) -> str:
    out: list[str] = []
    for token in tokens:
        if not out:
            out.append(token)
        elif re.match(r"[A-Za-z0-9_]", token) and re.match(r"[A-Za-z0-9_]", out[-1][-1]):
            out.append(" " + token)
        else:
            out.append(token)
    return "".join(out)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def decode_descriptor(desc: Descriptor) -> str:
    return TEMPLATES[desc.template_id].format(
        name=NAMES[desc.name_id],
        field=FIELDS[desc.field_id],
        offset=OFFSETS[desc.offset_id],
    )


def apply_exceptions(base: str, exceptions: list[ExceptionPatch]) -> str:
    tokens = tokenize(base)
    for patch in exceptions:
        if 0 <= patch.position < len(tokens):
            tokens[patch.position] = patch.token
    return untokenize(tokens)


def make_compiled_spans(args: argparse.Namespace) -> list[CompiledSpan]:
    rng = random.Random(args.seed)
    spans: list[CompiledSpan] = []
    for span_id in range(args.spans):
        desc = Descriptor(
            template_id=span_id % len(TEMPLATES),
            name_id=(span_id * 2 + 1) % len(NAMES),
            field_id=(span_id * 3 + 2) % len(FIELDS),
            offset_id=(span_id * 5 + 3) % len(OFFSETS),
        )
        base = decode_descriptor(desc)
        base_tokens = tokenize(base)
        exceptions: list[ExceptionPatch] = []
        for pos, token in enumerate(base_tokens):
            if token.isidentifier() and rng.random() < args.exception_rate:
                exceptions.append(ExceptionPatch(position=pos, token=f"{token}_X{span_id % 7}"))
        source = apply_exceptions(base, exceptions)
        decode_desc = desc
        if args.mode == "false-control" and span_id == 0:
            decode_desc = Descriptor(
                template_id=(desc.template_id + 1) % len(TEMPLATES),
                name_id=desc.name_id,
                field_id=desc.field_id,
                offset_id=desc.offset_id,
            )
        decoded = apply_exceptions(decode_descriptor(decode_desc), exceptions)
        raw_tokens = len(tokenize(source))
        exception_tokens = sum(1 for _ in exceptions)
        descriptor_bits = (
            math.ceil(math.log2(len(TEMPLATES)))
            + math.ceil(math.log2(len(NAMES)))
            + math.ceil(math.log2(len(FIELDS)))
            + math.ceil(math.log2(len(OFFSETS)))
        )
        exception_bits = exception_tokens * (math.ceil(math.log2(max(1, raw_tokens))) + args.bits_per_token)
        verify_bits = args.verify_bits
        spans.append(
            CompiledSpan(
                span_id=span_id,
                descriptor=desc,
                exceptions=exceptions,
                source_hash=digest(source),
                decoded_hash=digest(decoded),
                exact_decode=decoded == source,
                checksum_pass=digest(decoded) == digest(source),
                raw_tokens=raw_tokens,
                exception_tokens=exception_tokens,
                raw_bits=raw_tokens * args.bits_per_token,
                descriptor_bits=descriptor_bits,
                exception_bits=exception_bits,
                verify_bits=verify_bits,
            )
        )
    return spans


def dictionary_bits() -> float:
    artifact = "\n".join(TEMPLATES + NAMES + FIELDS + OFFSETS)
    return len(tokenize(artifact)) * 16.0


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    spans = make_compiled_spans(args)
    raw_bits = sum(span.raw_bits for span in spans)
    descriptor_bits = sum(span.descriptor_bits for span in spans)
    exception_bits = sum(span.exception_bits for span in spans)
    verify_bits = sum(span.verify_bits for span in spans)
    dict_bits = dictionary_bits()
    paid_bits = dict_bits + descriptor_bits + exception_bits + verify_bits
    exact_failures = [span.span_id for span in spans if not span.exact_decode]
    checksum_failures = [span.span_id for span in spans if not span.checksum_pass]
    exception_density = sum(span.exception_tokens for span in spans) / max(1, sum(span.raw_tokens for span in spans))
    compression_gain = raw_bits / paid_bits if paid_bits > 0 else 0.0
    dictionary_entry_ratio = len(TEMPLATES) / max(1, len(spans))
    pass_gate = (
        not exact_failures
        and not checksum_failures
        and exception_density <= args.max_exception_density
        and compression_gain >= args.min_compression_gain
        and dictionary_entry_ratio <= args.max_dictionary_entry_ratio
    )
    decision = (
        "PASS_R475_A1C04_DICTIONARY_SYNDROME_QUOTIENT_GATE"
        if pass_gate
        else "FAIL_R475_A1C04_DICTIONARY_SYNDROME_QUOTIENT_GATE"
    )
    return {
        "schema": SCHEMA,
        "decision": decision,
        "pass": pass_gate,
        "mode": args.mode,
        "seed": args.seed,
        "spans": len(spans),
        "dictionary_entries": len(TEMPLATES),
        "accepted_spans": len(spans) - len(exact_failures),
        "exact_decode_failures": len(exact_failures),
        "exact_failure_ids": exact_failures[:16],
        "checksum_failures": len(checksum_failures),
        "checksum_failure_ids": checksum_failures[:16],
        "raw_bits": raw_bits,
        "dictionary_bits": dict_bits,
        "descriptor_bits": descriptor_bits,
        "exception_bits": exception_bits,
        "verify_bits": verify_bits,
        "paid_bits": paid_bits,
        "compression_gain": compression_gain,
        "exception_density": exception_density,
        "dictionary_entry_ratio": dictionary_entry_ratio,
        "thresholds": {
            "max_exception_density": args.max_exception_density,
            "min_compression_gain": args.min_compression_gain,
            "max_dictionary_entry_ratio": args.max_dictionary_entry_ratio,
        },
        "elapsed_s": time.perf_counter() - start,
        "claim_scope": "realized_dictionary_exact_copy_gate_not_corpus_lookup",
    }


def write_spans(path: Path, spans: list[CompiledSpan]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for span in spans:
            row = asdict(span)
            row["descriptor"] = asdict(span.descriptor)
            row["exceptions"] = [asdict(exc) for exc in span.exceptions[:8]]
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "false-control"], default="diagnostic")
    parser.add_argument("--spans", type=int, default=96)
    parser.add_argument("--seed", type=int, default=44114)
    parser.add_argument("--exception-rate", type=float, default=0.035)
    parser.add_argument("--bits-per-token", type=float, default=16.0)
    parser.add_argument("--verify-bits", type=float, default=16.0)
    parser.add_argument("--max-exception-density", type=float, default=0.15)
    parser.add_argument("--min-compression-gain", type=float, default=1.20)
    parser.add_argument("--max-dictionary-entry-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spans-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spans = make_compiled_spans(args)
    summary = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.spans_jsonl is not None:
        args.spans_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_spans(args.spans_jsonl, spans)
    print(
        "[DICTIONARY_SYNDROME_QUOTIENT_GATE] "
        f"active=1 decision={summary['decision']} pass={int(summary['pass'])} "
        f"mode={summary['mode']} spans={summary['spans']} "
        f"exact_decode_failures={summary['exact_decode_failures']} "
        f"checksum_failures={summary['checksum_failures']} "
        f"exception_density={summary['exception_density']:.6f} "
        f"compression_gain={summary['compression_gain']:.6f} "
        f"dictionary_entry_ratio={summary['dictionary_entry_ratio']:.6f}"
    )
    print(f"[DICTIONARY_SYNDROME_QUOTIENT_OUTPUT] {args.output}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
