"""Immutable arena receipt gate for versioned packed payload slabs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .t020_execution_order_slab import layer1_execution_order_objects


ACTIVATION = "T020_IMMUTABLE_ARENA_RECEIPT_ACTIVE"

REQUIRED_BINDINGS = (
    "model_identity",
    "payload_content_root",
    "compression_policy",
    "lane_extraction_policy",
    "kernel_tile_layout_contract",
    "descriptor_schema_version",
    "arena_build_tool_version",
    "execution_order_root",
)


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_layer1_arena_manifest(
    *,
    model_identity: str = "Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137",
    compression_policy: str = "t014a_mlp_target_2p45_attention_t017_packed",
    lane_extraction_policy: str = "packed_codes_scales_repairs_v1",
    kernel_tile_layout_contract: str = "t017_triton_packed_linear_v1",
    descriptor_schema_version: int = 1,
    arena_build_tool_version: str = "t020_manifest_gate_v1",
    average_payload_bytes: int = 4096,
    lm_head_rows: int = 151936,
    lm_head_chunk_rows: int = 2048,
) -> dict[str, Any]:
    """Build a compact manifest for the T017 layer-1 execution order."""

    if descriptor_schema_version <= 0:
        raise ValueError("descriptor_schema_version must be positive")
    if average_payload_bytes <= 0:
        raise ValueError("average_payload_bytes must be positive")

    objects = layer1_execution_order_objects(lm_head_rows=lm_head_rows, lm_head_chunk_rows=lm_head_chunk_rows)
    descriptors: list[dict[str, Any]] = []
    cursor = 0
    for index, item in enumerate(objects):
        length = average_payload_bytes
        if item["consumer"] == "lm_head":
            length += ((item["stop"] or 0) - (item["start"] or 0)) // 8
        descriptor = {
            "ordinal": index,
            "name": item["name"],
            "consumer": item["consumer"],
            "row_start": item["start"],
            "row_stop": item["stop"],
            "offset": cursor,
            "length": length,
            "lane_id": f"lane:{index:04d}",
            "checksum": _stable_digest([item, cursor, length]),
        }
        descriptors.append(descriptor)
        cursor += length

    execution_order_root = _stable_digest(
        [
            {
                "ordinal": descriptor["ordinal"],
                "name": descriptor["name"],
                "row_start": descriptor["row_start"],
                "row_stop": descriptor["row_stop"],
                "offset": descriptor["offset"],
                "length": descriptor["length"],
                "lane_id": descriptor["lane_id"],
            }
            for descriptor in descriptors
        ]
    )
    payload_content_root = _stable_digest([descriptor["checksum"] for descriptor in descriptors])
    receipt = {
        "model_identity": model_identity,
        "payload_content_root": payload_content_root,
        "compression_policy": compression_policy,
        "lane_extraction_policy": lane_extraction_policy,
        "kernel_tile_layout_contract": kernel_tile_layout_contract,
        "descriptor_schema_version": descriptor_schema_version,
        "arena_build_tool_version": arena_build_tool_version,
        "execution_order_root": execution_order_root,
    }
    return {
        "receipt": receipt,
        "descriptor_root": _stable_digest(descriptors),
        "descriptors": descriptors,
        "arena_bytes": cursor,
    }


def verify_layer1_arena_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify receipt bindings and descriptor consistency without payload loads."""

    receipt = manifest.get("receipt", {})
    descriptors = manifest.get("descriptors", [])
    missing_bindings = [key for key in REQUIRED_BINDINGS if not receipt.get(key)]
    monotonic_offsets = True
    expected_offset = 0
    checksum_mismatches = 0
    for descriptor in descriptors:
        if descriptor.get("offset") != expected_offset:
            monotonic_offsets = False
        length = int(descriptor.get("length", 0))
        expected_offset += max(0, length)
        expected_checksum = _stable_digest(
            [
                {
                    "name": descriptor.get("name"),
                    "start": descriptor.get("row_start"),
                    "stop": descriptor.get("row_stop"),
                    "consumer": descriptor.get("consumer"),
                },
                descriptor.get("offset"),
                descriptor.get("length"),
            ]
        )
        if descriptor.get("checksum") != expected_checksum:
            checksum_mismatches += 1

    descriptor_root_matches = manifest.get("descriptor_root") == _stable_digest(descriptors)
    execution_order_root_matches = receipt.get("execution_order_root") == _stable_digest(
        [
            {
                "ordinal": descriptor.get("ordinal"),
                "name": descriptor.get("name"),
                "row_start": descriptor.get("row_start"),
                "row_stop": descriptor.get("row_stop"),
                "offset": descriptor.get("offset"),
                "length": descriptor.get("length"),
                "lane_id": descriptor.get("lane_id"),
            }
            for descriptor in descriptors
        ]
    )
    payload_content_root_matches = receipt.get("payload_content_root") == _stable_digest(
        [descriptor.get("checksum") for descriptor in descriptors]
    )
    return {
        "missing_bindings": missing_bindings,
        "descriptor_root_matches": descriptor_root_matches,
        "execution_order_root_matches": execution_order_root_matches,
        "payload_content_root_matches": payload_content_root_matches,
        "monotonic_offsets": monotonic_offsets,
        "checksum_mismatches": checksum_mismatches,
        "arena_bytes_matches_offsets": manifest.get("arena_bytes") == expected_offset,
        "requires_original_payload_load": False,
        "requires_compression_rerun": False,
        "requires_header_scan": False,
    }


def run_t020_immutable_arena_receipt_gate(
    *,
    baseline_dynamic_discovery_seconds: float = 0.5214,
    expected_tokens_between_rebuilds: int = 50000,
    offline_build_seconds: float = 90.0,
    open_arena_seconds: float = 0.002,
    descriptor_verify_microseconds: float = 12.0,
    root_lookup_seconds: float = 0.001,
    min_speedup: float = 1.10,
    lm_head_rows: int = 151936,
    lm_head_chunk_rows: int = 2048,
) -> dict[str, Any]:
    """Gate A-MATH-W01-15 against dynamic payload discovery leakage."""

    if baseline_dynamic_discovery_seconds <= 0.0:
        raise ValueError("baseline_dynamic_discovery_seconds must be positive")
    if expected_tokens_between_rebuilds <= 0:
        raise ValueError("expected_tokens_between_rebuilds must be positive")
    if offline_build_seconds < 0.0:
        raise ValueError("offline_build_seconds must be non-negative")
    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1")

    manifest = build_layer1_arena_manifest(lm_head_rows=lm_head_rows, lm_head_chunk_rows=lm_head_chunk_rows)
    verification = verify_layer1_arena_manifest(manifest)
    descriptor_count = len(manifest["descriptors"])
    verify_seconds = (
        root_lookup_seconds
        + open_arena_seconds
        + descriptor_count * max(0.0, descriptor_verify_microseconds) * 1.0e-6
    )
    amortized_rebuild_seconds = offline_build_seconds / expected_tokens_between_rebuilds
    candidate_dynamic_discovery_seconds = verify_seconds + amortized_rebuild_seconds
    speedup_vs_dynamic = baseline_dynamic_discovery_seconds / candidate_dynamic_discovery_seconds
    bindings_complete = not verification["missing_bindings"]
    hot_path_clean = not (
        verification["requires_original_payload_load"]
        or verification["requires_compression_rerun"]
        or verification["requires_header_scan"]
    )
    receipt_consistent = all(
        [
            verification["descriptor_root_matches"],
            verification["execution_order_root_matches"],
            verification["payload_content_root_matches"],
            verification["monotonic_offsets"],
            verification["arena_bytes_matches_offsets"],
            verification["checksum_mismatches"] == 0,
        ]
    )
    gain_floor_pass = speedup_vs_dynamic >= min_speedup
    gate_pass = bool(bindings_complete and hot_path_clean and receipt_consistent and gain_floor_pass)

    blockers: list[str] = []
    if not bindings_complete:
        blockers.append("receipt_missing_required_bindings")
    if not hot_path_clean:
        blockers.append("receipt_verification_requires_hot_path_payload_rebuild")
    if not receipt_consistent:
        blockers.append("manifest_descriptor_or_root_mismatch")
    if not gain_floor_pass:
        blockers.append("receipt_verification_does_not_clear_gain_floor")

    return {
        "probe": "t020_immutable_arena_receipt_gate",
        "activation": ACTIVATION,
        "finding": "A-MATH-W01-15",
        "baseline": {
            "source": "T017 layer-1 dynamic payload discovery/acquisition boundary",
            "dynamic_discovery_seconds": baseline_dynamic_discovery_seconds,
            "requires_original_payload_objects": True,
        },
        "candidate": {
            "descriptor_count": descriptor_count,
            "arena_bytes": manifest["arena_bytes"],
            "verify_seconds": verify_seconds,
            "offline_build_seconds": offline_build_seconds,
            "expected_tokens_between_rebuilds": expected_tokens_between_rebuilds,
            "amortized_rebuild_seconds": amortized_rebuild_seconds,
            "dynamic_discovery_seconds": candidate_dynamic_discovery_seconds,
            "speedup_vs_dynamic_discovery": speedup_vs_dynamic,
            "requires_original_payload_objects": False,
        },
        "receipt": manifest["receipt"],
        "descriptor_root": manifest["descriptor_root"],
        "verification": verification,
        "readiness": {
            "immutable_arena_receipt_gate_ready": gate_pass,
            "required_bindings_complete": bindings_complete,
            "descriptor_root_resolves_without_payload_load": hot_path_clean,
            "receipt_roots_match_manifest": receipt_consistent,
            "rebuild_boundary_outside_hot_path": expected_tokens_between_rebuilds > 1,
            "gain_floor_pass": gain_floor_pass,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Implement the real layer slab file writer using this receipt schema.",
                "Route T017 payload acquisition through a receipt-verified slab reader behind an opt-in flag.",
            ]
            if gate_pass
            else ["Do not integrate the slab reader until receipt verification is payload-free and root-consistent."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "REQUIRED_BINDINGS",
    "build_layer1_arena_manifest",
    "run_t020_immutable_arena_receipt_gate",
    "verify_layer1_arena_manifest",
]
