#!/usr/bin/env python3
"""Materialize R427 capability/artifact material into a v12 uint16 dataset.

The capability-material and artifact-shape gates produce real project-owned
objects, but the v12 trainer consumes a uint16 token stream. This bridge turns
those objects into a deterministic training corpus so an integrated R427 preset
can train on the material directly instead of stopping at standalone proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


SCHEMA = "r427_materialized_training_bin.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-jsonl", required=True, type=Path)
    parser.add_argument("--artifact-json", required=True, type=Path)
    parser.add_argument("--output-bin", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=65_536)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=42770)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
    return rows


def stable_token(text: str, vocab_size: int) -> int:
    if vocab_size <= 16:
        raise ValueError("vocab_size must leave room for reserved tokens")
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return 16 + (value % (vocab_size - 16))


def row_to_terms(row: dict[str, Any]) -> list[str]:
    proof = row.get("proof", {})
    proof_terms = []
    if isinstance(proof, dict):
        for key, value in sorted(proof.items()):
            proof_terms.append(f"proof:{key}:{json.dumps(value, sort_keys=True)}")
    return [
        "<row>",
        f"family:{row.get('family')}",
        f"concept:{row.get('concept_id')}",
        f"x:{row.get('x')}",
        f"y:{row.get('y')}",
        f"rotation:{row.get('rotation')}",
        f"cost:{row.get('cost')}",
        *proof_terms,
        "</row>",
    ]


def artifact_to_terms(artifact: dict[str, Any]) -> list[str]:
    terms = ["<artifact>"]
    for key, value in sorted(artifact.items()):
        terms.append(f"artifact:{key}:{json.dumps(value, sort_keys=True)}")
    terms.append("</artifact>")
    return terms


def terms_to_tokens(terms: list[str], vocab_size: int) -> list[int]:
    special = {
        "<bos>": 1,
        "<eos>": 2,
        "<row>": 3,
        "</row>": 4,
        "<artifact>": 5,
        "</artifact>": 6,
        "<pad>": 7,
    }
    tokens = []
    for term in terms:
        tokens.append(special.get(term, stable_token(term, vocab_size)))
    return tokens


def make_token_stream(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    *,
    target_tokens: int,
    vocab_size: int,
    seed: int,
) -> list[int]:
    if not rows:
        raise ValueError("material JSONL has no rows")
    base_terms = ["<bos>", f"schema:{SCHEMA}", f"seed:{seed}"]
    base_terms.extend(artifact_to_terms(artifact))
    for row in sorted(rows, key=lambda item: (str(item.get("family")), int(item.get("concept_id", 0)), int(item.get("rotation", 0)), int(item.get("x", 0)))):
        base_terms.extend(row_to_terms(row))
    base_terms.append("<eos>")
    base_tokens = terms_to_tokens(base_terms, vocab_size)
    if len(base_tokens) >= target_tokens:
        return base_tokens[:target_tokens]
    repeated: list[int] = []
    cursor = 0
    while len(repeated) < target_tokens:
        repeated.append(base_tokens[cursor % len(base_tokens)])
        cursor += 1
    return repeated


def write_uint16(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for token in tokens:
            handle.write(struct.pack("<H", token))


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.material_jsonl)
    artifact = json.loads(args.artifact_json.read_text(encoding="utf-8"))
    tokens = make_token_stream(
        rows,
        artifact,
        target_tokens=args.target_tokens,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    write_uint16(args.output_bin, tokens)
    digest = hashlib.sha256(args.output_bin.read_bytes()).hexdigest()
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("family"))
        family_counts[family] = family_counts.get(family, 0) + 1
    summary = {
        "schema": SCHEMA,
        "status": "ok",
        "activation_log": "[R427_MATERIALIZED_TRAINING_BIN] active=1 status=ok",
        "claim_scope": (
            "Deterministic bridge from R427 material/compiler artifacts into a "
            "v12 uint16 training stream. This validates trainability of the "
            "material path, not final quality or speed."
        ),
        "material_jsonl": str(args.material_jsonl),
        "artifact_json": str(args.artifact_json),
        "output_bin": str(args.output_bin),
        "token_count": len(tokens),
        "byte_size": args.output_bin.stat().st_size,
        "vocab_size": args.vocab_size,
        "sha256": digest,
        "material_rows": len(rows),
        "family_counts": family_counts,
        "artifact_keys": sorted(artifact.keys()),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[R427_MATERIALIZED_TRAINING_BIN] "
        f"active=1 status=ok tokens={len(tokens)} bytes={summary['byte_size']} "
        f"rows={len(rows)} output={args.output_bin} summary={args.summary_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
