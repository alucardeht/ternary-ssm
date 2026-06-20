"""Run the first synthetic probe for the T001 transformer residency line."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_residency import (
    TileReceipt,
    gate_cliff_certificate,
    margin_certificate,
    qwen3_32b_shape,
    residual_collapse_report,
    run_default_gate_suite,
    run_t002_kv_margin_sweep,
    run_t003_receipt_kv_gate,
    run_t004_real_qwen_bridge,
    run_t005_real_weight_sentinel,
    run_t006_real_residency_block_smoke,
    run_t007_real_layer_residency_smoke,
    run_t008_compressed_generation_smoke,
    run_t009_t013_runtime_scaler,
    run_t014_denominator_tightening,
    run_t015_packed_qwen_inference,
    run_t016_persistent_packed_cache,
    run_t017_kernel_qwen_inference,
    run_t017_packed_linear_kernel_gate,
    run_t018_resident_payload_qwen_inference,
    run_t019_binary_payload_qwen_inference,
    run_t020_acceptance_weighted_kv_materialization_gate,
    run_t020_affine_packed_view_gate,
    run_t020_basis_shared_kv_transform_gate,
    run_t020_conversion_time_arena_compiler_gate,
    run_t020_cumulative_execution_shape_gate,
    run_t020_device_credit_flow_gate,
    run_t020_expected_accepted_span_gate,
    run_t020_execution_order_slab_gate,
    run_t020_immutable_arena_receipt_gate,
    run_t020_packed_work_coalescing_gate,
    run_t020_prefix_transaction_ledger_gate,
    run_t020_recompute_on_reject_gate,
    run_t020_ring_exact_tail_receipts_gate,
    run_t021_cumulative_decoder_runtime_path_gate,
    run_t021_page_activation_certificate_gate,
    run_t021_qwen_runtime,
    run_t021_runtime_obligation_measure_gate,
    run_t021_transaction_receipt_gate,
    run_t022_qwen_runtime,
    run_t023_qwen_runtime,
    run_t024_qwen_runtime,
    run_t025_qwen_runtime,
    run_t026_qwen_runtime,
    run_t027_qwen_runtime,
    run_t028_native_generation_speed,
    run_t029_speed_gpu_efficiency,
    run_t030_speed_gpu_efficiency,
    run_t031_macro_architecture_stack,
    run_t034_gpu_real_runtime_gates,
    run_t035_row_successor_gate,
    run_t035_row_successor_runtime_path,
    run_t035_runtime_guardrail_gate,
    run_t036_a05_block_verifier_gate,
    run_t037_direct_denominator_gate,
    run_t037_exact_suffix_prefill_gate,
    run_t038_kernel_body_roofline_gate,
    run_t038_representative_map_gate,
    run_t038b_coverage_denominator_gate,
    run_t039_a01_exact_map_object_ledger_gate,
    run_t039_a02_semantic_layout_ledger_gate,
    run_t039b_a01_non_row_exact_object_extraction_ledger,
    run_t039b_a02_gauge_sat_canonical_tail_ledger,
    run_t039_real_map_object_audit,
)


DEFAULT_OUT = ROOT / ".work" / "jobs" / "t001-transformer-residency-probe"


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_int_csv(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from exc


def _parse_float_csv(value: str) -> tuple[float, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one number")
    try:
        return tuple(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number list: {value}") from exc


def _parse_str_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one name")
    return items


def _runtime_receipt_console_summary(receipt: dict | None) -> dict | None:
    if receipt is None:
        return None
    cache_stats = receipt.get("cache_stats") or {}
    return {
        "generated_token_ids": receipt.get("generated_token_ids"),
        "elapsed_seconds": receipt.get("elapsed_seconds"),
        "tokens_per_second": receipt.get("tokens_per_second"),
        "kernel_seconds": receipt.get("kernel_seconds"),
        "kernel_timing_mode": receipt.get("kernel_timing_mode"),
        "defer_kernel_synchronization": receipt.get("defer_kernel_synchronization"),
        "gpu_argmax_head": receipt.get("gpu_argmax_head"),
        "packed_linear_block_m": receipt.get("packed_linear_block_m"),
        "packed_linear_block_c": receipt.get("packed_linear_block_c"),
        "packed_linear_block_r": receipt.get("packed_linear_block_r"),
        "fused_mlp_gate_up_kernel": receipt.get("fused_mlp_gate_up_kernel"),
        "fused_qkv_compact_projection_kernel": receipt.get("fused_qkv_compact_projection_kernel"),
        "trit_code_group_size": receipt.get("trit_code_group_size"),
        "peak_cuda_allocated_gib": receipt.get("peak_cuda_allocated_gib"),
        "peak_cuda_reserved_gib": receipt.get("peak_cuda_reserved_gib"),
        "peak_process_working_set_gib": receipt.get("peak_process_working_set_gib"),
        "process_private_usage_gib": receipt.get("process_private_usage_gib"),
        "preload_seconds": receipt.get("preload_seconds"),
        "hot_payload_load_events": receipt.get("hot_payload_load_events"),
        "hot_payload_load_seconds": receipt.get("hot_payload_load_seconds"),
        "device_payload_descriptor_count": len(receipt.get("device_payload_descriptor_table") or {}),
        "cache_stats": {
            "device_resident_bytes": cache_stats.get("device_resident_bytes"),
            "device_resident_gib": cache_stats.get("device_resident_gib"),
            "device_resident_hits": cache_stats.get("device_resident_hits"),
            "device_resident_stores": cache_stats.get("device_resident_stores"),
            "device_arena_hits": cache_stats.get("device_arena_hits"),
            "device_arena_misses": cache_stats.get("device_arena_misses"),
            "trit_payloads_loaded": cache_stats.get("trit_payloads_loaded"),
            "trit_cache_gib_read": cache_stats.get("trit_cache_gib_read"),
        },
    }


def _random_orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def make_synthetic_case(seed: int, samples: int, intermediate: int, hidden: int) -> dict:
    rng = np.random.default_rng(seed)
    visible_rank = max(4, hidden // 16)
    down_basis = _random_orthonormal(rng, intermediate, hidden)
    down_projection = down_basis / np.sqrt(hidden)

    visible_subspace = _random_orthonormal(rng, intermediate, visible_rank)
    invisible_noise = rng.normal(size=(samples, intermediate))
    invisible_noise -= invisible_noise @ down_basis @ down_basis.T
    hidden_residual = (
        0.20 * rng.normal(size=(samples, visible_rank)) @ visible_subspace.T
        + 1.00 * invisible_noise / np.sqrt(intermediate)
    )

    return {
        "hidden_residual": hidden_residual,
        "down_projection": down_projection,
        "visible_rank": visible_rank,
    }


def run_probe(args: argparse.Namespace) -> dict:
    case = make_synthetic_case(args.seed, args.samples, args.intermediate, args.hidden)
    collapse = residual_collapse_report(
        case["hidden_residual"],
        case["down_projection"],
        max_visible_energy_ratio=args.max_visible_energy_ratio,
        max_visible_rank_ratio=args.max_visible_rank_ratio,
    )

    tile_elements = args.tile_k * args.tile_n
    receipts = {
        "q2_5_with_residual": TileReceipt(
            dense_elements=tile_elements,
            payload_bits_per_element=2.5,
            metadata_bytes=args.metadata_bytes,
            scale_bytes=args.scale_bytes,
            correction_bytes=args.correction_bytes,
            scratch_bytes=args.scratch_bytes,
            launch_fragmentation_bytes=args.launch_fragmentation_bytes,
        ).as_dict(),
        "q4_baseline": TileReceipt(
            dense_elements=tile_elements,
            payload_bits_per_element=4.0,
            metadata_bytes=args.scale_bytes,
            scale_bytes=args.scale_bytes,
        ).as_dict(),
    }

    logits = np.array([1.0, 0.72, 0.2, -0.5], dtype=np.float64)
    margin = margin_certificate(logits, error_radius=args.logit_error_radius)

    rng = np.random.default_rng(args.seed + 1)
    gate_values = rng.normal(loc=0.0, scale=1.0, size=(args.samples, args.intermediate))
    gate_cert = gate_cliff_certificate(
        gate_values,
        residual_bound=args.gate_residual_bound,
        margin=args.gate_margin,
        min_accepted_fraction=args.min_gate_accepted_fraction,
    )
    gate_suite = run_default_gate_suite(args.seed)
    shape = qwen3_32b_shape()
    uniform_effective_bits = receipts["q2_5_with_residual"]["effective_bits_per_element"]
    qwen_projection = shape.residency_projection(
        candidate_bits=uniform_effective_bits,
        overhead_gib=args.runtime_overhead_gib,
        q4_overhead_gib=args.runtime_overhead_gib,
    )
    component_projection = shape.component_residency_projection(
        mlp_bits=args.mlp_effective_bits,
        attention_bits=args.attention_effective_bits,
        embedding_bits=args.embedding_effective_bits,
        output_head_bits=args.output_head_effective_bits,
        overhead_gib=args.runtime_overhead_gib,
        q4_overhead_gib=args.runtime_overhead_gib,
    )
    synthetic_decode_throughput = shape.synthetic_decode_throughput_projection(
        uniform_bits=uniform_effective_bits,
        mlp_bits=args.mlp_effective_bits,
        attention_bits=args.attention_effective_bits,
        embedding_bits=args.embedding_effective_bits,
        output_head_bits=args.output_head_effective_bits,
        bandwidth_gib_per_second=args.synthetic_bandwidth_gib_per_second,
        weight_streams_per_token=args.weight_streams_per_token,
        kv_cache_gib_per_token=args.kv_cache_gib_per_token,
    )

    return {
        "probe": "t001_transformer_residency_probe",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": vars(args),
        "residual_collapse": collapse.as_dict(),
        "tile_receipts": receipts,
        "margin_certificate": margin.as_dict(),
        "gate_cliff_certificate": gate_cert.as_dict(),
        "gate_suite": gate_suite.as_dict(),
        "qwen3_32b_projection": qwen_projection,
        "qwen3_32b_component_projection": component_projection,
        "synthetic_decode_throughput": synthetic_decode_throughput,
        "overall": {
            "passes_residual_collapse": collapse.passes_energy_gate and collapse.passes_rank_gate,
            "passes_tile_receipt": bool(receipts["q2_5_with_residual"]["passes_10pct_saved_gate"]),
            "passes_margin_certificate": margin.accepted,
            "passes_gate_cliff_certificate": gate_cert.accepted,
            "passes_gate_suite_controls": gate_suite.matched,
            "candidate_fits_12gib_projection": bool(qwen_projection["candidate_fits_12gib"]),
            "component_stack_fits_12gib_projection": bool(component_projection["candidate_fits_12gib"]),
        },
    }


def write_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T001 Transformer Residency Probe",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Overall",
        "",
    ]
    for key, value in result["overall"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Key Numbers",
            "",
            f"- residual energy ratio: `{result['residual_collapse']['energy_ratio']:.6f}`",
            f"- residual rank ratio: `{result['residual_collapse']['rank_ratio']:.6f}`",
            "- q2.5 plus residual effective bits: "
            f"`{result['tile_receipts']['q2_5_with_residual']['effective_bits_per_element']:.3f}`",
            "- q2.5 plus residual compression vs fp16: "
            f"`{result['tile_receipts']['q2_5_with_residual']['compression_ratio_vs_fp16']:.3f}x`",
            "- projected Qwen3-32B candidate footprint: "
            f"`{result['qwen3_32b_projection']['candidate_estimated_gib']:.3f} GiB`",
            "- projected Qwen3-32B Q4 footprint: "
            f"`{result['qwen3_32b_projection']['q4_estimated_gib']:.3f} GiB`",
            "- projected candidate saves vs Q4: "
            f"`{result['qwen3_32b_projection']['candidate_saves_gib_vs_q4']:.3f} GiB`",
            "- projected component-stack footprint: "
            f"`{result['qwen3_32b_component_projection']['candidate_estimated_gib']:.3f} GiB`",
            "- projected component-stack effective bits: "
            f"`{result['qwen3_32b_component_projection']['effective_bits_per_weight']:.3f}`",
            "- projected component-stack saves vs uniform probe: "
            f"`{result['qwen3_32b_projection']['candidate_estimated_gib'] - result['qwen3_32b_component_projection']['candidate_estimated_gib']:.3f} GiB`",
            "- synthetic decode proxy bandwidth: "
            f"`{result['synthetic_decode_throughput']['bandwidth_gib_per_second']:.1f} GiB/s`",
            "- synthetic Q4 decode throughput proxy: "
            f"`{result['synthetic_decode_throughput']['q4_tokens_per_second']:.2f} tok/s`",
            "- synthetic uniform-candidate decode throughput proxy: "
            f"`{result['synthetic_decode_throughput']['uniform_tokens_per_second']:.2f} tok/s`",
            "- synthetic component-stack decode throughput proxy: "
            f"`{result['synthetic_decode_throughput']['component_tokens_per_second']:.2f} tok/s`",
            "- synthetic component-stack speedup vs Q4: "
            f"`{result['synthetic_decode_throughput']['component_speedup_vs_q4']:.3f}x`",
            "- synthetic component-stack speedup vs uniform probe: "
            f"`{result['synthetic_decode_throughput']['component_speedup_vs_uniform']:.3f}x`",
            f"- logit margin safety ratio: `{result['margin_certificate']['safety_ratio']:.3f}`",
            f"- gate accepted fraction: `{result['gate_cliff_certificate']['accepted_fraction']:.3f}`",
            f"- gate suite unexpected controls: `{result['gate_suite']['unexpected_count']}`",
            "",
            "The tok/s proxy is a synthetic memory-bound decode estimate, not a model benchmark.",
            "This synthetic probe is a wiring and gate sanity check, not a model result.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t002_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T002 KV Margin Sweep",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Overall",
        "",
    ]
    for key, value in result["overall"].items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.6f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
        ]
    )
    for key, value in result["candidate_summary"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## KV Denominator",
            "",
            f"- FP16 KV bytes/token: `{result['kv_denominator']['fp16_bytes_per_token']}`",
            f"- FP16 KV KiB/token: `{result['kv_denominator']['fp16_kib_per_token']:.3f}`",
        ]
    )
    for context, gib in result["kv_denominator"]["fp16_gib_by_context"].items():
        lines.append(f"- FP16 KV at `{context}` tokens: `{gib:.3f} GiB`")

    lines.extend(
        [
            "",
            "## Scenario Comparison",
            "",
            "| scenario | ctx | total GiB | slack GiB | saved vs before | tok/s proxy | speedup | fits |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in result["scenarios"]:
        lines.append(
            "| {name} | {context_tokens} | {total_gib:.3f} | {slack_gib:.3f} | "
            "{vram_saved_vs_baseline_gib:.3f} | {synthetic_tokens_per_second:.2f} | "
            "{speedup_vs_baseline:.3f}x | {fits_12gib} |".format(**scenario)
        )

    lines.extend(
        [
            "",
            "## Validations",
            "",
        ]
    )
    for key, value in result["validations"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "This is a synthetic denominator and wiring gate. It is not a real Qwen3 VRAM run or a measured model tok/s benchmark.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t003_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T003 Receipt And KV Precision Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Overall",
        "",
    ]
    for key, value in result["overall"].items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.6f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Receipt",
            "",
            f"- audit hash: `{result['residency_receipt']['audit_hash']}`",
            f"- total without safety: `{result['residency_receipt']['total_without_safety_gib']:.3f} GiB`",
            f"- total with safety: `{result['residency_receipt']['total_with_safety_gib']:.3f} GiB`",
            f"- slack without safety: `{result['residency_receipt']['slack_without_safety_gib']:.3f} GiB`",
            f"- slack with safety: `{result['residency_receipt']['slack_with_safety_gib']:.3f} GiB`",
            "",
            "## KV Policy",
            "",
            f"- context tokens: `{result['kv_policy']['context_tokens']}`",
            f"- exact recent tokens: `{result['kv_policy']['exact_recent_tokens']}`",
            f"- old key/value bits: `{result['kv_policy']['old_key_bits']:.2f}` / `{result['kv_policy']['old_value_bits']:.2f}`",
            f"- KV residency: `{result['kv_policy']['kv_gib']:.3f} GiB`",
            f"- residual error bound: `{result['kv_policy']['residual_error_bound']:.3f}`",
            f"- residual budget: `{result['kv_policy']['residual_budget']:.3f}`",
            "",
            "## Tok/s Proxy",
            "",
            f"- baseline proxy: `{result['tok_s_predictor']['baseline']['tokens_per_second']:.2f} tok/s`",
            f"- T003 proxy: `{result['tok_s_predictor']['t003']['tokens_per_second']:.2f} tok/s`",
            f"- speedup: `{result['tok_s_predictor']['speedup_vs_baseline']:.3f}x`",
            "",
            "## Validations",
            "",
        ]
    )
    for key, value in result["validations"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "This is a synthetic receipt and denominator gate. It is not a real Qwen3-32B VRAM, quality, or generation benchmark.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t004_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    env = result["environment"]
    inventory = result["weight_inventory"]
    config = result["config"]
    lines = [
        "# T004 Real Qwen Runtime Bridge",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Config",
            "",
            f"- model id: `{result['model_id']}`",
            f"- source: `{config['source']}`",
            f"- source path: `{config['source_path']}`",
            f"- shape match: `{config['matches_project_qwen3_32b_shape']}`",
            f"- layers / hidden / intermediate: `{config['layers']} / {config['hidden_size']} / {config['intermediate_size']}`",
            f"- heads / KV heads / head dim: `{config['attention_heads']} / {config['kv_heads']} / {config['head_dim']}`",
            f"- max positions: `{config['max_position_embeddings']}`",
            f"- config sha256: `{config['config_sha256']}`",
            "",
            "## Environment",
            "",
            f"- packages: `{env['packages']}`",
            f"- torch CUDA: `{env['torch_cuda']}`",
            f"- nvidia-smi: `{env['nvidia_smi']}`",
            f"- disk free near model: `{env['disk_free_gib_near_model']}` GiB",
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- total file GiB: `{inventory['total_file_gib']:.3f}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
            f"- largest tensor: `{inventory['largest_tensor_name']}` / `{inventory['largest_tensor_gib']:.3f} GiB`",
            f"- inventory error: `{inventory['error']}`",
            "",
            "## T003 Reference",
            "",
            f"- total with safety: `{result['t003_reference']['total_with_safety_gib']:.3f} GiB`",
            f"- slack with safety: `{result['t003_reference']['slack_with_safety_gib']:.3f} GiB`",
            f"- tok/s proxy: `{result['t003_reference']['tok_s_proxy']:.2f}`",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    next_actions = result["next_actions"] or ["No next action required by the bridge preflight."]
    for action in next_actions:
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "This bridge does not load full model weights and does not claim real Qwen generation throughput.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t005_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    lines = [
        "# T005 Real Weight Sentinel",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
            "",
            "## Tensor Sentinels",
            "",
        ]
    )
    for item in result["sentinels"]:
        lines.append(
            f"- `{item['name']}` from `{item['shard']}`: shape `{item['shape']}`, "
            f"sample `{item['sample_shape']}`, finite `{item['finite_ratio']:.6f}`, "
            f"nonzero `{item['nonzero_ratio']:.6f}`, abs mean `{item['abs_mean']:.6g}`, "
            f"std `{item['std']:.6g}`, sha256 `{item['sample_sha256'][:16]}`"
        )
    if result["missing_tensors"]:
        lines.extend(["", "## Missing Requested Tensors", ""])
        for name in result["missing_tensors"]:
            lines.append(f"- `{name}`")
    lines.extend(["", "## Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "This sentinel reads tiny real tensor prefixes only. It does not load or generate from the full model."])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t006_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    lines = [
        "# T006 Real Residency Block Smoke",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
            "",
            "## Compressed Blocks",
            "",
        ]
    )
    for item in result["blocks"]:
        lines.append(
            f"- `{item['tensor']}` policy `{item['policy']['name']}`: effective bits "
            f"`{item['effective_bits_per_weight']:.3f}`, Q4 saved "
            f"`{item['q4_savings_fraction']:.3f}`, weight rRMSE "
            f"`{item['weight_relative_rmse']:.3f}`, matmul rRMSE "
            f"`{item['matmul_relative_rmse']:.3f}`, pass `{item['passes']}`"
        )
    lines.extend(["", "## Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This is the first project-owned compressed-residency smoke on real Qwen weights. It validates real tensor access, block compression, reconstruction, and local matmul error, but it is not full-token generation yet.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t007_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    layer = result["layer_receipt"]
    lines = [
        "# T007 Real Layer Residency Smoke",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
        ]
    )
    if layer is not None:
        lines.extend(
            [
                "",
                "## Layer Receipt",
                "",
                f"- layer: `{layer['layer_index']}`",
                f"- effective bits per weight: `{layer['effective_bits_per_weight']:.4f}`",
                f"- Q4 payload savings: `{layer['q4_savings_fraction']:.4f}`",
                f"- attention rRMSE: `{layer['attention_relative_rmse']:.4f}`",
                f"- MLP rRMSE: `{layer['mlp_relative_rmse']:.4f}`",
                f"- layer output rRMSE: `{layer['layer_relative_rmse']:.4f}`",
                f"- pass: `{layer['passes']}`",
                "",
                "## Tensor Receipts",
                "",
            ]
        )
        for item in result["tensor_receipts"]:
            lines.append(
                f"- `{item['name']}` policy `{item['policy']['name']}`: bits "
                f"`{item['effective_bits_per_weight']:.3f}`, Q4 saved "
                f"`{item['q4_savings_fraction']:.3f}`, weight rRMSE "
                f"`{item['weight_relative_rmse']:.3f}`"
            )
    lines.extend(["", "## Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This is a sliced real-layer smoke. It exercises attention and MLP projections from one real Qwen layer through the project codec, but it is not a full-model generation benchmark.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t008_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    generation = result["generation_receipt"]
    lines = [
        "# T008 Compressed Generation Smoke",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
        ]
    )
    if generation is not None:
        lines.extend(
            [
                "",
                "## Generation Receipt",
                "",
                f"- layers: `{generation['layer_indices']}`",
                f"- hidden cols: `{generation['hidden_cols']}`",
                f"- attention width: `{generation['attention_width']}`",
                f"- mlp width: `{generation['mlp_width']}`",
                f"- vocab rows: `{generation['vocab_rows']}`",
                f"- prompt tokens: `{generation['prompt_tokens']}`",
                f"- compressed generated tokens: `{generation['generated_tokens']}`",
                f"- exact top-1 tokens: `{generation['exact_top1_tokens']}`",
                f"- top-1 agreement: `{generation['top1_agreement']:.4f}`",
                f"- avg logits rRMSE: `{generation['avg_logits_relative_rmse']:.4f}`",
                f"- max logits rRMSE: `{generation['max_logits_relative_rmse']:.4f}`",
                f"- effective bits per weight: `{generation['effective_bits_per_weight']:.4f}`",
                f"- Q4 payload savings: `{generation['q4_savings_fraction']:.4f}`",
                f"- compressed seconds: `{generation['compressed_seconds']:.6f}`",
                f"- sliced compressed tok/s: `{generation['compressed_tokens_per_second']:.4f}`",
                f"- receipt sha256: `{generation['generation_sha256']}`",
                f"- pass: `{generation['passes']}`",
            ]
        )
    lines.extend(["", "## Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This is a sliced multi-layer micro-generation smoke over real Qwen tensors. It validates compressed-token execution through embeddings, selected layers, and lm_head, but it is not a full hidden-width or full-model Qwen generation benchmark.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t009_t013_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    t009 = result["t009_full_width_layer"]
    t010 = result["t010_multilayer_scale"]
    t011 = result["t011_tokenizer_prompt"]
    t012 = result["t012_kv_cache_ledger"]
    t013 = result["t013_full_model_equivalent_pricing"]
    lines = [
        "# T009-T013 Qwen Runtime Scaler",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
            "",
            "## T009 Full-Width Layer",
            "",
            f"- activation: `{t009['activation']}`",
            f"- layer index: `{t009['layer_index']}`",
            f"- projection count: `{t009['projection_count']}`",
            f"- effective bits per weight: `{t009['effective_bits_per_weight']:.4f}`",
            f"- Q4 payload savings: `{t009['q4_savings_fraction']:.4f}`",
            f"- average projection rRMSE: `{t009['avg_projection_relative_rmse']:.4f}`",
            f"- compressed projection seconds: `{t009['compressed_seconds']:.6f}`",
            f"- pass: `{t009['passes']}`",
            "",
            "## T010 Multi-Layer Scale",
            "",
            f"- activation: `{t010['activation']}`",
            f"- layer indices: `{t010['layer_indices']}`",
            f"- row limit: `{t010['row_limit']}`",
            f"- projection count: `{t010['projection_count']}`",
            f"- effective bits per weight: `{t010['effective_bits_per_weight']:.4f}`",
            f"- Q4 payload savings: `{t010['q4_savings_fraction']:.4f}`",
            f"- average projection rRMSE: `{t010['avg_projection_relative_rmse']:.4f}`",
            f"- pass: `{t010['passes']}`",
            "",
            "## T011 Tokenizer Prompt",
            "",
            f"- activation: `{t011['activation']}`",
            f"- token count: `{t011['token_count']}`",
            f"- token ids: `{t011['token_ids']}`",
            f"- pass: `{t011['passes']}`",
            "",
            "## T012 KV Cache Ledger",
            "",
            f"- activation: `{t012['activation']}`",
            f"- fp16 KiB/token: `{t012['fp16_kib_per_token']:.4f}`",
            f"- KV effective bits: `{t012['kv_effective_bits']}`",
            f"- exact recent tokens: `{t012['exact_recent_tokens']}`",
            f"- pass: `{t012['passes']}`",
            "",
            "## T013 Full-Model-Equivalent Pricing",
            "",
            f"- activation: `{t013['activation']}`",
            f"- candidate total with KV GiB: `{t013['candidate_total_with_kv_gib']:.4f}`",
            f"- Q4 total with FP16 KV GiB: `{t013['q4_total_with_fp16_kv_gib']:.4f}`",
            f"- candidate saves GiB: `{t013['candidate_saves_vs_q4_total_gib']:.4f}`",
            f"- candidate fits 12 GiB: `{t013['candidate_fits_12gib']}`",
            f"- full-model-equivalent tok/s: `{t013['full_model_equivalent_tokens_per_second']:.4f}`",
            f"- Q4 FP16-KV tok/s: `{t013['q4_fp16kv_tokens_per_second']:.4f}`",
            f"- speedup vs Q4 FP16-KV: `{t013['speedup_vs_q4_fp16kv']:.4f}`",
            f"- pass: `{t013['passes']}`",
            f"- pricing blockers: `{t013.get('pricing_blockers', [])}`",
            "",
            "## Execution Blockers",
            "",
        ]
    )
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Pricing Blockers", ""])
    pricing_blockers = result.get("pricing_blockers") or t013.get("pricing_blockers") or ["none"]
    for blocker in pricing_blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This is a staged runtime scaler. T009 and T010 use real Qwen projections; T011 validates tokenizer wiring; T012 prices KV-cache memory; T013 is full-model-equivalent pricing, not measured full-model generation.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t014_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    mlp = result["mlp_projection_gate"]
    packed = result["packed_arena_receipt"]
    baseline = result["baseline"]
    lines = [
        "# T014 Denominator Tightening Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
            "",
            "## Baseline",
            "",
            f"- current T013 total with KV GiB: `{baseline['total_with_kv_gib']:.4f}`",
            f"- current T013 tok/s proxy: `{baseline['tokens_per_second_proxy']:.4f}`",
            f"- Q4 total with FP16 KV GiB: `{baseline['q4_total_with_fp16_kv_gib']:.4f}`",
            f"- Q4 FP16-KV tok/s proxy: `{baseline['q4_fp16kv_tokens_per_second']:.4f}`",
            "",
            "## MLP Target-Bits Projection Gate",
            "",
            f"- activation: `{mlp['activation']}`",
            f"- layer index: `{mlp['layer_index']}`",
            f"- row limit: `{mlp['row_limit']}`",
            f"- projection count: `{mlp['projection_count']}`",
            f"- target MLP bits: `{mlp['target_mlp_bits']:.4f}`",
            f"- effective bits per weight: `{mlp['effective_bits_per_weight']:.4f}`",
            f"- average projection rRMSE: `{mlp['avg_projection_relative_rmse']:.4f}`",
            f"- target bits pass: `{mlp['target_bits_pass']}`",
            f"- pass: `{mlp['passes']}`",
            "",
            "## Packed Arena Receipt",
            "",
            f"- activation: `{packed['activation']}`",
            f"- current overhead GiB: `{packed['current_overhead_gib']:.4f}`",
            f"- packed overhead GiB: `{packed['packed_overhead_gib']:.4f}`",
            f"- saved GiB: `{packed['saved_gib']:.4f}`",
            f"- pass: `{packed['passes']}`",
            "",
            "## Candidate Table",
            "",
            "| candidate | total GiB | slack GiB | saved vs T013 | tok/s proxy | speedup vs Q4 | status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in result["candidate_table"]:
        lines.append(
            "| {name} | {total_with_kv_gib:.4f} | {slack_vs_12gib:.4f} | "
            "{saves_vs_current_t013_gib:.4f} | {tokens_per_second_proxy:.4f} | "
            "{speedup_vs_q4_fp16kv:.4f}x | {status} |".format(**candidate)
        )
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate validates denominator fit and real-weight MLP projection behavior. It does not claim full all-layer Qwen text inference or measured full-model tok/s.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t015_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    receipt = result["receipt"]
    lines = [
        "# T015 Packed Qwen Streamed Inference Smoke",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
        ]
    )
    if receipt is not None:
        lines.extend(
            [
                "",
                "## Inference Receipt",
                "",
                f"- prompt: `{receipt['prompt_text']}`",
                f"- tokenizer source: `{receipt['tokenizer_source']}`",
                f"- prompt token count: `{receipt['prompt_token_count']}`",
                f"- generated token ids: `{receipt['generated_token_ids']}`",
                f"- generated text: `{receipt['generated_text']}`",
                f"- requested layers: `{receipt['requested_layers']}`",
                f"- completed layers: `{receipt['completed_layers']}`",
                f"- device: `{receipt['device']}`",
                f"- elapsed seconds: `{receipt['elapsed_seconds']:.4f}`",
                f"- measured tok/s: `{receipt['tokens_per_second']:.6f}`",
                f"- peak CUDA allocated GiB: `{receipt['peak_cuda_allocated_gib']}`",
                f"- peak CUDA reserved GiB: `{receipt['peak_cuda_reserved_gib']}`",
                f"- T014-A denominator total GiB: `{receipt['denominator_total_gib']:.4f}`",
                f"- T014-A denominator slack GiB: `{receipt['denominator_slack_gib']:.4f}`",
                f"- output sha256: `{receipt['output_sha256']}`",
                f"- pass: `{receipt['passes']}`",
            ]
        )
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This smoke executes token generation through the project compressed-weight path. Streamed on-the-fly reconstruction is functional validation, not optimized packed GPU residency or final tok/s.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t016_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    receipt = result["receipt"]
    lines = [
        "# T016 Persistent Packed Cache Smoke",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
        ]
    )
    if receipt is not None:
        cache_stats = receipt["cache_stats"]
        lines.extend(
            [
                "",
                "## Inference Receipt",
                "",
                f"- prompt: `{receipt['prompt_text']}`",
                f"- tokenizer source: `{receipt['tokenizer_source']}`",
                f"- prompt token count: `{receipt['prompt_token_count']}`",
                f"- generated token ids: `{receipt['generated_token_ids']}`",
                f"- generated text: `{receipt['generated_text']}`",
                f"- requested layers: `{receipt['requested_layers']}`",
                f"- completed layers: `{receipt['completed_layers']}`",
                f"- device: `{receipt['device']}`",
                f"- elapsed seconds: `{receipt['elapsed_seconds']:.4f}`",
                f"- measured tok/s: `{receipt['tokens_per_second']:.6f}`",
                f"- peak CUDA allocated GiB: `{receipt['peak_cuda_allocated_gib']}`",
                f"- peak CUDA reserved GiB: `{receipt['peak_cuda_reserved_gib']}`",
                f"- T014-A denominator total GiB: `{receipt['denominator_total_gib']:.4f}`",
                f"- T014-A denominator slack GiB: `{receipt['denominator_slack_gib']:.4f}`",
                f"- output sha256: `{receipt['output_sha256']}`",
                f"- pass: `{receipt['passes']}`",
                "",
                "## Cache Stats",
                "",
                f"- cache dir: `{cache_stats['cache_dir']}`",
                f"- tensors built: `{cache_stats['cache_build_count']}`",
                f"- tensors loaded: `{cache_stats['cache_hit_count']}`",
                f"- cache GiB written: `{cache_stats['cache_gib_written']:.6f}`",
                f"- cache GiB read: `{cache_stats['cache_gib_read']:.6f}`",
                f"- pack seconds: `{cache_stats['pack_seconds']:.4f}`",
                f"- load seconds: `{cache_stats['load_seconds']:.4f}`",
                f"- reconstruct seconds: `{cache_stats['reconstruct_seconds']:.4f}`",
            ]
        )
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This smoke validates the pre-kernel cache boundary: compressed tensors persist between runs, then materialize dense weights for matmul. It is not the fused packed-kernel path.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t017_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    receipt = result["receipt"]
    readiness = result["readiness"]
    lines = [
        "# T017 Packed Linear Kernel Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    if receipt is not None:
        lines.extend(
            [
                "",
                "## Kernel Receipt",
                "",
                f"- backend: `{receipt['backend']}`",
                f"- rows: `{receipt['rows']}`",
                f"- cols: `{receipt['cols']}`",
                f"- tokens: `{receipt['tokens']}`",
                f"- repair count: `{receipt['repair_count']}`",
                f"- iterations: `{receipt['iterations']}`",
                f"- max abs error: `{receipt['max_abs_error']:.8f}`",
                f"- relative RMSE: `{receipt['relative_rmse']:.8f}`",
                f"- reference seconds/iter: `{receipt['reference_seconds_per_iter']:.8f}`",
                f"- kernel seconds/iter: `{receipt['kernel_seconds_per_iter']:.8f}`",
                f"- speedup vs reconstruct+matmul: `{receipt['speedup_vs_reconstruct_matmul']:.4f}x`",
                f"- output sha256: `{receipt['output_sha256']}`",
                f"- pass: `{receipt['passes']}`",
            ]
        )
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate tests the generic packed linear kernel boundary only. It does not claim all-layer Qwen throughput until the kernel is integrated into the runtime path.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t017_qwen_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    readiness = result["readiness"]
    inventory = result["weight_inventory"]
    receipt = result["receipt"]
    if result["probe"].startswith("t030_"):
        report_title = "# T030 Speed And GPU Efficiency Smoke"
    elif result["probe"].startswith("t029_"):
        report_title = "# T029 Speed And GPU Efficiency Smoke"
    elif result["probe"].startswith("t028_"):
        report_title = "# T028 Native Generation Speed Smoke"
    elif result["probe"].startswith("t027_"):
        report_title = "# T027 Generation Speed Runtime Smoke"
    elif result["probe"].startswith("t022_"):
        report_title = "# T022 Qwen Runtime Smoke"
    elif result["probe"].startswith("t021_"):
        report_title = "# T021 Qwen Runtime Smoke"
    elif result["probe"].startswith("t018_"):
        report_title = "# T018 Resident Payload Qwen Inference Smoke"
    elif result["probe"].startswith("t019_"):
        report_title = "# T019 Binary Payload Qwen Inference Smoke"
    elif result["probe"].startswith("t026_"):
        report_title = "# T026 GPU-Owned Qwen Runtime Smoke"
    elif result["probe"].startswith("t025_"):
        report_title = "# T025 GPU Resident Qwen Runtime Smoke"
    else:
        report_title = "# T017 Kernel Qwen Inference Smoke"
    lines = [
        report_title,
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Local Weights",
            "",
            f"- model dir: `{inventory['model_dir']}`",
            f"- local weights present: `{inventory['local_weights_present']}`",
            f"- safetensors shards: `{inventory['shard_count']}`",
            f"- tensor count: `{inventory['tensor_count']}`",
            f"- tensor payload GiB: `{inventory['tensor_payload_gib']:.3f}`",
        ]
    )
    if receipt is not None:
        cache_stats = receipt["cache_stats"]
        lines.extend(
            [
                "",
                "## Inference Receipt",
                "",
                f"- prompt: `{receipt['prompt_text']}`",
                f"- tokenizer source: `{receipt['tokenizer_source']}`",
                f"- prompt token count: `{receipt['prompt_token_count']}`",
                f"- generated token ids: `{receipt['generated_token_ids']}`",
                f"- generated text: `{receipt['generated_text']}`",
                f"- requested layers: `{receipt['requested_layers']}`",
                f"- completed layers: `{receipt['completed_layers']}`",
                f"- device: `{receipt['device']}`",
                f"- elapsed seconds: `{receipt['elapsed_seconds']:.4f}`",
                f"- measured tok/s: `{receipt['tokens_per_second']:.6f}`",
                f"- kernel seconds: `{receipt['kernel_seconds']:.4f}`",
                f"- resident payload cache: `{receipt.get('resident_payload_cache', False)}`",
                f"- preload payloads: `{receipt.get('preload_payloads', False)}`",
                f"- preload seconds: `{receipt.get('preload_seconds', 0.0):.4f}`",
                f"- trit packed codes: `{receipt.get('trit_packed_codes', False)}`",
                f"- fail-closed device payloads: `{receipt.get('fail_closed_device_payloads', False)}`",
                f"- hot payload load events: `{receipt.get('hot_payload_load_events', 0)}`",
                f"- hot payload load seconds: `{receipt.get('hot_payload_load_seconds', 0.0):.4f}`",
                f"- device credit flow: `{bool(receipt.get('device_credit_flow'))}`",
                f"- packed work coalescing: `{bool(receipt.get('packed_work_coalescing'))}`",
                f"- peak CUDA allocated GiB: `{receipt['peak_cuda_allocated_gib']}`",
                f"- peak CUDA reserved GiB: `{receipt['peak_cuda_reserved_gib']}`",
                f"- peak process working set GiB: `{receipt.get('peak_process_working_set_gib')}`",
                f"- process working set GiB: `{receipt.get('process_working_set_gib')}`",
                f"- process private usage GiB: `{receipt.get('process_private_usage_gib')}`",
                f"- T014-A denominator total GiB: `{receipt['denominator_total_gib']:.4f}`",
                f"- T014-A denominator slack GiB: `{receipt['denominator_slack_gib']:.4f}`",
                f"- output sha256: `{receipt['output_sha256']}`",
                f"- pass: `{receipt['passes']}`",
                "",
                "## Cache Stats",
                "",
                f"- cache dir: `{cache_stats['cache_dir']}`",
                f"- tensors built: `{cache_stats['cache_build_count']}`",
                f"- tensors loaded: `{cache_stats['cache_hit_count']}`",
                f"- resident stores: `{cache_stats.get('resident_stores', 0)}`",
                f"- resident hits: `{cache_stats.get('resident_hits', 0)}`",
                f"- resident GiB: `{cache_stats.get('resident_gib', 0.0):.6f}`",
                f"- device resident stores: `{cache_stats.get('device_resident_stores', 0)}`",
                f"- device resident hits: `{cache_stats.get('device_resident_hits', 0)}`",
                f"- device arena hits: `{cache_stats.get('device_arena_hits', 0)}`",
                f"- device arena misses: `{cache_stats.get('device_arena_misses', 0)}`",
                f"- device resident GiB: `{cache_stats.get('device_resident_gib', 0.0):.6f}`",
                f"- binary payloads written: `{cache_stats.get('binary_payloads_written', 0)}`",
                f"- binary payloads loaded: `{cache_stats.get('binary_payloads_loaded', 0)}`",
                f"- binary cache GiB written: `{cache_stats.get('binary_cache_gib_written', 0.0):.6f}`",
                f"- binary cache GiB read: `{cache_stats.get('binary_cache_gib_read', 0.0):.6f}`",
                f"- binary write seconds: `{cache_stats.get('binary_write_seconds', 0.0):.4f}`",
                f"- binary load seconds: `{cache_stats.get('binary_load_seconds', 0.0):.4f}`",
                f"- cache GiB written: `{cache_stats['cache_gib_written']:.6f}`",
                f"- cache GiB read: `{cache_stats['cache_gib_read']:.6f}`",
                f"- pack seconds: `{cache_stats['pack_seconds']:.4f}`",
                f"- load seconds: `{cache_stats['load_seconds']:.4f}`",
                f"- reconstruct seconds: `{cache_stats['reconstruct_seconds']:.4f}`",
            ]
        )
        if result["probe"].startswith("t021_"):
            t021_gate = result.get("t021_cumulative_gate", {})
            gate_readiness = t021_gate.get("readiness") or {}
            gate_candidate = t021_gate.get("candidate") or {}
            lines.extend(
                [
                    "",
                    "## T021 Runtime Gate",
                    "",
                    f"- gate activation: `{t021_gate.get('activation')}`",
                    f"- gate ready: `{gate_readiness.get('cumulative_decoder_runtime_path_gate_ready')}`",
                    f"- runtime ready: `{readiness.get('t021_qwen_runtime_ready')}`",
                    f"- non-overlap projection: `{gate_candidate.get('non_overlapping_denominator_projection')}`",
                ]
            )
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    note = "This smoke uses Triton packed-linear kernels for the Qwen projections and lm_head chunks."
    if result["probe"].startswith(("t025_", "t026_", "t027_", "t028_", "t029_", "t030_")):
        note += " The GPU-resident variants require zero hot payload-load events before readiness can pass."
    else:
        note += " It still loads packed payload files from the T016 cache on the hot path."
    lines.extend(["", note])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_device_credit_flow_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Device Credit Flow Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| total seconds/token | `{baseline['total_seconds']:.4f}` | "
            f"`{candidate['total_seconds']:.4f}` | projected static gate |"
        ),
        (
            f"| tokens/s | `{baseline['tokens_per_second']:.6f}` | "
            f"`{candidate['tokens_per_second']:.6f}` | projected static gate |"
        ),
        (
            f"| live credit memory GiB | `{baseline['denominator_total_gib']:.4f}` denominator | "
            f"`+{candidate['live_credit_gib']:.4f}` credit window | 12 GiB constraint |"
        ),
        (
            f"| speedup | `1.0000x` | `{candidate['speedup_vs_baseline']:.4f}x` | "
            f"{'passes' if readiness['gain_floor_pass'] else 'below floor'} |"
        ),
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Resource Credits", ""])
    for row in result["resource_table"]:
        lines.append(f"- `{row['resource']}`: `{row}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "This gate is diagnostic only. It prices the B-MATH-W02-09 credit-flow invariant before any production scheduler integration."])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_packed_work_coalescing_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    traffic = result["traffic"]
    readiness = result["readiness"]
    lines = [
        "# T020 Packed Work Coalescing Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## One-Block Result",
        "",
        f"- legal boundaries: `{result['legal_boundaries']}`",
        f"- legal outputs per block: `{readiness['legal_outputs_per_block']}`",
        f"- fused residual adds across priced depth: `{traffic['fused_residual_adds']}`",
        f"- deleted dense traffic MiB: `{traffic['deleted_dense_mib']:.6f}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Boundary Map", ""])
    for row in result["one_block_boundary_map"]:
        lines.append(f"- `{row['producer']}`: `{row}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate proves the legal residual-destination survivors before enabling opt-in packed-kernel residual accumulation.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_execution_order_slab_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Execution-Order Slab Arena Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        f"| acquisition objects | `{baseline['objects']}` | `{candidate['objects']}` | static gate |",
        (
            f"| acquisition seconds | `{baseline['acquisition_seconds']:.6f}` | "
            f"`{candidate['acquisition_seconds']:.6f}` | modeled acquisition-only |"
        ),
        f"| acquisition speedup | `1.0000x` | `{candidate['acquisition_speedup']:.4f}x` | modeled acquisition-only |",
        f"| offset entries | `0` | `{candidate['offset_entries']}` | compact offset walk |",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "This gate prices object acquisition only. It does not claim measured Qwen runtime tok/s."])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t021_report(result: dict, out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    if "baseline" in result and "candidate" in result:
        lines.extend(["", "## Baseline Versus Candidate", ""])
        baseline = result["baseline"]
        candidate = result["candidate"]
        lines.append(f"- baseline policy: `{baseline.get('policy')}`")
        lines.append(f"- candidate policy: `{candidate.get('policy')}`")
        for key, value in candidate.items():
            if key == "policy":
                continue
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"- candidate `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    next_actions = result.get("next_actions") or ["none"]
    for action in next_actions:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("This is a T021 gate result. It does not claim measured production tok/s.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t031_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    candidate = result["candidate"]
    speed_projection = candidate["speed_projection"]
    memory_projection = candidate["memory_projection"]
    receipts = candidate["algorithmic_receipts"]
    runtime_comparison = candidate.get("runtime_comparison") or {}
    runtime_evidence = candidate.get("runtime_integration_evidence") or {}
    t033_greedy = candidate.get("t033_greedy_lm_head_output_contract") or {}
    t033_descriptor_gate = candidate.get("t033_descriptor_denominator_gate") or {}
    lines = [
        "# T031 Macro Architecture Stack",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Baseline Versus Candidate",
            "",
            f"- baseline tok/s: `{result['baseline']['speed_tokens_per_second']}`",
            f"- projected tok/s: `{speed_projection['projected_tokens_per_second']}`",
            f"- projected speedup: `{speed_projection['combined_projected_speedup']}`",
            f"- baseline peak CUDA GiB: `{result['baseline']['speed_peak_cuda_gib']}`",
            f"- projected peak CUDA GiB: `{memory_projection['projected_peak_cuda_gib']}`",
            f"- projected GPU saved GiB: `{memory_projection['projected_gpu_saved_gib']}`",
            f"- mechanism count: `{candidate['mechanism_count']}`",
            f"- algorithmic receipts ready: `{sum(1 for receipt in receipts.values() if receipt['ready'])}/{len(receipts)}`",
            f"- measured runtime tok/s: `{runtime_comparison.get('measured_tokens_per_second')}`",
            f"- measured runtime peak CUDA GiB: `{runtime_comparison.get('measured_peak_cuda_allocated_gib')}`",
            f"- measured runtime resident CUDA GiB: `{runtime_comparison.get('measured_resident_cuda_gib')}`",
            "",
            "## Mechanisms",
            "",
        ]
    )
    for mechanism in candidate["mechanisms"]:
        lines.append(
            f"- `{mechanism['id']}` {mechanism['name']} "
            f"({mechanism['kind']}, {mechanism['denominator']})"
        )
    lines.extend(["", "## Algorithmic Receipts", ""])
    for mechanism_id, receipt in sorted(receipts.items()):
        lines.append(f"- `{mechanism_id}` ready=`{receipt['ready']}` claim=`{receipt['claim']}`")
    lines.extend(["", "## Real Runtime Integration", ""])
    for mechanism_id, evidence in sorted(runtime_evidence.items()):
        if not mechanism_id.startswith("T031-"):
            continue
        lines.append(
            f"- `{mechanism_id}` real_runtime_path=`{evidence['real_runtime_path']}` "
            f"evidence=`{evidence['evidence']}`"
        )
    lines.extend(["", "## T033 Runtime Closure Gates", ""])
    lines.append(
        f"- `T033-A-V01` greedy LM-head contract ready=`{t033_greedy.get('ready')}` "
        f"full_logits_materialized=`{t033_greedy.get('full_logits_materialized')}` "
        f"full_logits_returned_to_host=`{t033_greedy.get('full_logits_returned_to_host')}`"
    )
    lines.append(
        f"- `T033-C` descriptor denominator gate ready=`{t033_descriptor_gate.get('ready')}` "
        f"positive_gate_count=`{t033_descriptor_gate.get('positive_gate_count')}` "
        f"descriptor_count=`{t033_descriptor_gate.get('descriptor_count')}`"
    )
    for gate_name, gate in sorted((t033_descriptor_gate.get("gates") or {}).items()):
        lines.append(f"  - `{gate_name}` ready=`{gate.get('ready')}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result.get("next_actions") or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append(
        "This T031 report executes the cumulative real T030 runtime path when invoked from the CLI. "
        "Projected tok/s remains planning evidence; real tok/s is the measured runtime field above."
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t034_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    comparison = result["t034_comparison"]
    gates = result["t034_gates"]
    runtime = result.get("runtime") or {}
    receipt = runtime.get("receipt") or {}
    lines = [
        "# T034 GPU-Real Runtime Gates",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Measured Comparison",
        "",
        f"- baseline source: `{comparison['baseline_source']}`",
        f"- comparison scope: `{comparison['comparison_scope']}`",
        f"- baseline tok/s: `{comparison['baseline_tokens_per_second']}`",
        f"- measured tok/s: `{comparison['measured_tokens_per_second']}`",
        f"- measured speedup: `{comparison['measured_speedup_vs_r459_t033']}`",
        f"- baseline peak CUDA GiB: `{comparison['baseline_peak_cuda_gib']}`",
        f"- measured peak CUDA GiB: `{comparison['measured_peak_cuda_gib']}`",
        f"- baseline resident CUDA GiB: `{comparison['baseline_resident_cuda_gib']}`",
        f"- measured resident CUDA GiB: `{comparison['measured_resident_cuda_gib']}`",
        f"- production tok/s claim allowed: `{comparison['production_tok_s_claim_allowed']}`",
        "",
        "## Gates",
        "",
    ]
    for name, gate in gates.items():
        gate = gate or {}
        lines.append(f"- `{name}`: status=`{gate.get('status')}` ready=`{gate.get('ready_for_promotion')}`")
    lines.extend(
        [
            "",
            "## Runtime Receipt",
            "",
            f"- generated token ids: `{receipt.get('generated_token_ids')}`",
            f"- hot payload load events: `{receipt.get('hot_payload_load_events')}`",
            f"- device payload cache: `{receipt.get('device_payload_cache')}`",
            f"- t034 splice: `{receipt.get('t034_device_argmax_embedding_splice')}`",
            "",
            "## Readiness",
            "",
        ]
    )
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result.get("next_actions") or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t035_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    scratch = result["scratch_contract"]
    projection = result["projection"]
    lines = [
        "# T035 Runtime Guardrail Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Scope",
        "",
        "- Implements `C02-V02` and `A01-V04` as a prerequisite guardrail.",
        "- This is not a standalone production speed claim.",
        "- Full64 projections use R452/T028, not one-layer R459/T033.",
        "",
        "## Scratch And Peak Contract",
        "",
        f"- max live scratch GiB: `{scratch['max_live_scratch_gib']}`",
        f"- candidate peak CUDA GiB: `{scratch['candidate_peak_cuda_gib']}`",
        f"- peak delta versus R452 GiB: `{scratch['peak_delta_vs_r452_gib']}`",
        f"- peak cap GiB: `{scratch['peak_cap_gib']}`",
        f"- hidden dense live GiB: `{scratch['hidden_dense_live_gib']}`",
        "",
        "## Projection Boundary",
        "",
        f"- full64 before tok/s: `{projection['full64_before_tokens_per_second']}`",
        f"- guardrail-only tok/s band: `{projection['guardrail_only_tokens_per_second_band']}`",
        f"- production speed claim allowed: `{projection['production_speed_claim_allowed']}`",
        f"- ratio required for 15 tok/s: `{projection['target_15_tok_s_ratio_required']}`",
        f"- ratio required for 40 tok/s: `{projection['target_40_tok_s_ratio_required']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t035_row_successor_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    candidate = result["candidate"]
    census = result["allocation_census"]
    projection = result["projection"]
    lines = [
        "# T035 Row Successor Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Candidate",
        "",
        f"- device: `{result['shape']['device']}`",
        f"- token id: `{candidate['token_id']}`",
        f"- reference token id: `{candidate['reference_token_id']}`",
        f"- exact token match: `{candidate['exact_token_match']}`",
        f"- exact embedding match: `{candidate['exact_embedding_match']}`",
        f"- next embedding ready before host token: `{candidate['next_embedding_ready_before_host_token']}`",
        f"- local gate tok/s: `{candidate['tokens_per_second_local_gate']}`",
        f"- measured peak CUDA allocated GiB: `{candidate['measured_peak_cuda_allocated_gib']}`",
        "",
        "## Allocation Census",
        "",
        f"- candidate full logits buffer GiB: `{census['candidate_full_logits_buffer_gib']}`",
        f"- candidate table-sized scratch: `{census['candidate_table_sized_scratch']}`",
        f"- scaled full64 candidate peak GiB: `{census['scaled_full64_candidate_peak_gib']}`",
        f"- full64 peak cap GiB: `{census['full64_peak_cap_gib']}`",
        "",
        "## Projection Boundary",
        "",
        f"- full64 before tok/s: `{projection['full64_before_tokens_per_second']}`",
        f"- row-successor-only tok/s band: `{projection['row_successor_only_tokens_per_second_band']}`",
        f"- production speed claim allowed: `{projection['production_speed_claim_allowed']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t035_row_successor_runtime_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    comparison = result["comparison"]
    projection = result["projection"]
    lines = [
        "# T035 Row Successor Runtime Path",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Runtime Comparison",
        "",
        f"- scope: `{comparison['scope']}`",
        f"- requested layers: `{comparison['requested_layers']}`",
        f"- completed layers: `{comparison['completed_layers']}`",
        f"- measured tok/s: `{comparison['measured_tokens_per_second']}`",
        f"- baseline source: `{comparison['baseline_source']}`",
        f"- baseline tok/s: `{comparison['baseline_tokens_per_second']}`",
        f"- measured speedup vs comparable baseline: `{comparison['measured_speedup_vs_comparable_baseline']}`",
        f"- measured peak CUDA GiB: `{comparison['measured_peak_cuda_gib']}`",
        f"- measured reserved CUDA GiB: `{comparison['measured_reserved_cuda_gib']}`",
        f"- measured resident CUDA GiB: `{comparison['measured_resident_cuda_gib']}`",
        f"- active peak cap GiB: `{comparison['active_peak_cap_gib']}`",
        f"- production full64 speed claim allowed: `{comparison['production_full64_speed_claim_allowed']}`",
        "",
        "## Projection Boundary",
        "",
        f"- full64 before tok/s: `{projection['full64_before_tokens_per_second']}`",
        f"- row-successor-only full64 band: `{projection['row_successor_only_full64_band']}`",
        f"- ratio required for 15 tok/s after this gate: `{projection['target_15_tok_s_ratio_required_after_this_gate']}`",
        f"- ratio required for 40 tok/s after this gate: `{projection['target_40_tok_s_ratio_required_after_this_gate']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t036_block_verifier_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T036 A05-06 Block-Verifier Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline: `{result['baseline']['tokens_per_second']}` tok/s",
        f"- baseline seconds/token: `{result['baseline']['seconds_per_token']}`",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        f"- target seconds/token: `{result['target']['seconds_per_token']}`",
        "",
        "## Current Executable Pricing",
        "",
        "| block | accepted | verifier mode | seconds/accepted token | tok/s | target pass |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for row in result["priced_blocks"]:
        lines.append(
            f"| `{row['block_size']}` | `{row['accepted_tokens']}` | `{row['mode']}` | "
            f"`{row['seconds_per_accepted_token']:.9f}` | `{row['tokens_per_second']:.6f}` | "
            f"`{row['target_pass']}` |"
        )
    lines.extend(
        [
            "",
            "## Conditional T036 Projection",
            "",
            "| block | accepted | projected cost multiplier | seconds/accepted token | tok/s | target pass |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in result["conditional_projection_blocks"]:
        lines.append(
            f"| `{row['block_size']}` | `{row['accepted_tokens']}` | "
            f"`{row['cost_multiplier_vs_r452_token']:.6f}` | "
            f"`{row['seconds_per_accepted_token']:.9f}` | `{row['tokens_per_second']:.6f}` | "
            f"`{row['target_pass']}` |"
        )
    lines.extend(["", "## Memory Contract", ""])
    for key, value in result["memory_contract"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t037_exact_suffix_prefill_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T037 Exact Suffix-Prefill Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline: `{result['baseline']['tokens_per_second']}` tok/s",
        f"- baseline seconds/token: `{result['baseline']['seconds_per_token']}`",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        f"- target seconds/token: `{result['target']['seconds_per_token']}`",
        "",
        "## Exactness Census",
        "",
        "| block | cases | committed states match serial | first mismatch indices match | toy elapsed seconds |",
        "|---:|---:|---|---|---:|",
    ]
    for row in result["exactness_census"]:
        lines.append(
            f"| `{row['block_size']}` | `{row['case_count']}` | "
            f"`{row['all_committed_states_match_serial']}` | "
            f"`{row['all_first_mismatch_indices_match']}` | "
            f"`{row['toy_elapsed_seconds']:.9f}` |"
        )
    lines.extend(
        [
            "",
            "## Full64 Pricing",
            "",
            "| block | accepted | timing source | suffix seconds | seconds/accepted token | tok/s | target pass |",
            "|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in result["pricing"]:
        suffix_seconds = "missing" if row["suffix_prefill_seconds"] is None else f"{row['suffix_prefill_seconds']:.9f}"
        seconds_per_token = (
            "missing"
            if row["seconds_per_accepted_token"] is None
            else f"{row['seconds_per_accepted_token']:.9f}"
        )
        tokens_per_second = "missing" if row["tokens_per_second"] is None else f"{row['tokens_per_second']:.6f}"
        lines.append(
            f"| `{row['block_size']}` | `{row['accepted_tokens']}` | `{row['timing_source']}` | "
            f"`{suffix_seconds}` | `{seconds_per_token}` | `{tokens_per_second}` | "
            f"`{row['target_pass']}` |"
        )
    lines.extend(["", "## Memory Contract", ""])
    for key, value in result["memory_contract"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t037_direct_denominator_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T037 Direct Denominator Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline: `{result['baseline']['tokens_per_second']}` tok/s",
        f"- baseline seconds/token: `{result['baseline']['seconds_per_token']}`",
        f"- baseline kernel seconds/token: `{result['baseline']['kernel_seconds_per_token']}`",
        f"- baseline non-kernel seconds/token: `{result['baseline']['non_kernel_seconds_per_token']}`",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        f"- target seconds/token: `{result['target']['seconds_per_token']}`",
        "",
        "## Scenarios",
        "",
        "| name | kernel s/tok | non-kernel s/tok | total s/tok | tok/s | peak CUDA GiB | target pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["scenarios"]:
        lines.append(
            f"| `{row['name']}` | `{row['kernel_seconds_per_token']:.9f}` | "
            f"`{row['non_kernel_seconds_per_token']:.9f}` | "
            f"`{row['total_seconds_per_token']:.9f}` | `{row['tokens_per_second']:.6f}` | "
            f"`{row['peak_cuda_gib']:.6f}` | `{row['target_pass']}` |"
        )
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t038_kernel_body_roofline_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T038 Kernel/Body Roofline Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline: `{result['baseline']['tokens_per_second']}` tok/s",
        f"- baseline seconds/token: `{result['baseline']['seconds_per_token']}`",
        f"- baseline kernel seconds/token: `{result['baseline']['kernel_seconds_per_token']}`",
        f"- baseline resident payload: `{result['baseline']['device_resident_gib']}` GiB",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        f"- target seconds/token: `{result['target']['seconds_per_token']}`",
        "",
        "## Roofline",
        "",
    ]
    for key, value in result["roofline"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Scenarios",
            "",
            "| name | hot bytes GiB | bandwidth GiB/s | byte s/tok | total s/tok | tok/s | target pass |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in result["scenarios"]:
        lines.append(
            f"| `{row['name']}` | `{row['hot_bytes_gib']:.6f}` | "
            f"`{row['sustained_bandwidth_gib_s']:.6f}` | "
            f"`{row['byte_seconds_per_token']:.9f}` | "
            f"`{row['total_seconds_per_token']:.9f}` | `{row['tokens_per_second']:.6f}` | "
            f"`{row['target_pass']}` |"
        )
    lines.extend(["", "## Format Audit", ""])
    for key, value in result["format_audit"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Layout Audit", ""])
    for key, value in result["layout_audit"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t038_representative_map_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T038 Representative Map Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline kernel seconds/token: `{result['baseline']['kernel_seconds_per_token']}`",
        f"- required kernel speedup: `{result['baseline']['required_kernel_speedup_if_control_zero']}`",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        "",
        "## Representative Map",
        "",
    ]
    for key, value in result["representative_map"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t038b_coverage_denominator_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T038B Coverage Denominator Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline kernel seconds/token: `{result['baseline']['kernel_seconds_per_token']}`",
        f"- required kernel speedup: `{result['target']['required_kernel_speedup_if_control_zero']}`",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        "",
        "## Coverage Model",
        "",
    ]
    for key, value in result["coverage_model"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Components",
            "",
            "| name | body fraction | speedup | denominator contribution | accelerated |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in result["components"]:
        lines.append(
            f"| `{row['name']}` | `{row['body_fraction']}` | "
            f"`{row['speedup_vs_current_component']}` | "
            f"`{row['harmonic_denominator_contribution']}` | `{row['accelerated']}` |"
        )
    lines.extend(["", "## Exactness Audit", ""])
    for key, value in result["exactness_audit"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t039_a01_exact_map_object_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T039 A01 Exact Map Object Ledger Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline tok/s: `{result['baseline']['current_tokens_per_second']}`",
        f"- baseline kernel seconds/token: `{result['baseline']['kernel_seconds_per_token']}`",
        f"- required kernel speedup: `{result['target']['required_kernel_speedup_if_control_zero']}`",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        "",
        "## Coverage Model",
        "",
    ]
    for key, value in result["coverage_model"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Components",
            "",
            "| name | body fraction | speedup | denominator contribution |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["components"]:
        lines.append(
            f"| `{row['name']}` | `{row['body_fraction']}` | "
            f"`{row['speedup_vs_current_component']}` | "
            f"`{row['harmonic_denominator_contribution']}` |"
        )
    lines.extend(["", "## Object Audit", ""])
    for key, value in result["object_audit"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t039_a02_semantic_layout_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T039 A02 Semantic Layout Ledger Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline And Target",
        "",
        f"- baseline tok/s: `{result['baseline']['current_tokens_per_second']}`",
        f"- baseline kernel seconds/token: `{result['baseline']['kernel_seconds_per_token']}`",
        f"- required kernel speedup: `{result['target']['required_kernel_speedup_if_control_zero']}`",
        f"- target: `{result['target']['tokens_per_second']}` tok/s",
        "",
        "## Coverage Model",
        "",
    ]
    for key, value in result["coverage_model"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Components",
            "",
            "| name | body fraction | speedup | denominator contribution |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["components"]:
        lines.append(
            f"| `{row['name']}` | `{row['body_fraction']}` | "
            f"`{row['speedup_vs_current_component']}` | "
            f"`{row['harmonic_denominator_contribution']}` |"
        )
    lines.extend(["", "## Layout Audit", ""])
    for key, value in result["layout_audit"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t039_real_map_object_audit_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T039 Real Map Object Audit",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Audited Blocks", ""])
    lines.extend(
        [
            "| tensor | rows | cols | unique rows | quotient speedup | exact |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in result["audited_blocks"]:
        lines.append(
            f"| `{row['tensor']}` | `{row['rows']}` | `{row['cols']}` | "
            f"`{row['unique_rows']}` | `{row['row_quotient_speedup']}` | "
            f"`{row['exact_reconstruction']}` |"
        )
    lines.extend(["", "## Ledger Summary", ""])
    ledger = result["ledger"]
    for key, value in ledger["coverage_model"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_immutable_arena_receipt_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Immutable Arena Receipt Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| dynamic discovery seconds | `{baseline['dynamic_discovery_seconds']:.6f}` | "
            f"`{candidate['dynamic_discovery_seconds']:.6f}` | modeled receipt verification |"
        ),
        (
            f"| descriptor count | original payload objects | `{candidate['descriptor_count']}` | "
            "manifest root |"
        ),
        (
            f"| speedup vs dynamic discovery | `1.0000x` | "
            f"`{candidate['speedup_vs_dynamic_discovery']:.4f}x` | modeled receipt gate |"
        ),
        (
            f"| original payload objects required | `{baseline['requires_original_payload_objects']}` | "
            f"`{candidate['requires_original_payload_objects']}` | hot-path receipt boundary |"
        ),
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Receipt Roots", ""])
    lines.append(f"- descriptor root: `{result['descriptor_root']}`")
    lines.append(f"- execution order root: `{result['receipt']['execution_order_root']}`")
    lines.append(f"- payload content root: `{result['receipt']['payload_content_root']}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate proves the manifest can verify arena identity without loading original payload objects.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_conversion_time_arena_compiler_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Conversion-Time Arena Compiler Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| visible interpretation denominator | `{baseline['visible_denominator_seconds']:.6f}` | "
            f"`{baseline['visible_denominator_seconds'] - candidate['deleted_denominator_seconds']:.6f}` | "
            "modeled compiler gate |"
        ),
        (
            f"| deleted denominator seconds | `0.000000` | "
            f"`{candidate['deleted_denominator_seconds']:.6f}` | compile-time artifact |"
        ),
        f"| saved fraction | `0.0000` | `{candidate['saved_fraction']:.4f}` | modeled compiler gate |",
        (
            f"| runtime interpreter required | `{baseline['runtime_interpreter_required']}` | "
            f"`{candidate['runtime_interpreter_required']}` | compiled views |"
        ),
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Descriptor Equality", ""])
    for key, value in result["descriptor_equality"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate proves compile-time descriptor equality and dependency classification only.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_affine_packed_view_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Affine Packed View Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| materialization required | `{baseline['materialization_required']}` | "
            f"`{candidate['materialization_required']}` | affine view |"
        ),
        (
            f"| net deleted seconds | `0.000000` | "
            f"`{candidate['net_deleted_seconds']:.6f}` | modeled materialization deletion |"
        ),
        f"| saved fraction | `0.0000` | `{candidate['saved_fraction']:.4f}` | modeled affine gate |",
        (
            f"| speedup vs visible denominator | `1.0000x` | "
            f"`{candidate['speedup_vs_visible_denominator']:.4f}x` | modeled affine gate |"
        ),
        "",
        "## Equality",
        "",
    ]
    for key, value in result["equality"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Readiness", ""])
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate proves affine packed-view equality on a synthetic exhaustive rectangle only.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_recompute_on_reject_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Recompute-On-Reject Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        f"| branch policy cost | `{baseline['eager_cost']:.6f}` | `{candidate['policy_cost']:.6f}` | modeled branch tree |",
        f"| deleted rejected-branch cost | `0.000000` | `{candidate['deleted_cost']:.6f}` | modeled branch tree |",
        f"| speedup vs eager KV writes | `1.0000x` | `{candidate['speedup_vs_eager']:.4f}x` | modeled branch tree |",
        f"| stored branch positions | `{candidate['total_positions']}` | `{candidate['stored_positions']}` | store/defer policy |",
        f"| deferred branch positions | `0` | `{candidate['deferred_positions']}` | exact recompute receipts |",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate prices speculative KV store/defer policy only; it does not run a decode engine.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_expected_accepted_span_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Expected-Accepted-Span Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        f"| selected span length | `{baseline['span_length']}` | `{candidate['span_length']}` | high-confidence profile |",
        (
            f"| expected accepted tokens | `{baseline['expected_accepted_tokens']:.6f}` | "
            f"`{candidate['expected_accepted_tokens']:.6f}` | denominator table |"
        ),
        (
            f"| cost per accepted token | `{baseline['cost_per_accepted_token']:.6f}` | "
            f"`{candidate['cost_per_accepted_token']:.6f}` | denominator table |"
        ),
        (
            f"| speedup vs one-token denominator | `1.0000x` | "
            f"`{candidate['speedup_vs_one_token']:.4f}x` | modeled span table |"
        ),
        "",
        "## Profile Choices",
        "",
        "| profile | selected b | R(1) | R(selected) | speedup | legal |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for profile in result["profiles"]:
        lines.append(
            f"| `{profile['profile']}` | `{profile['selected']['span_length']}` | "
            f"`{profile['baseline']['cost_per_accepted_token']:.6f}` | "
            f"`{profile['selected']['cost_per_accepted_token']:.6f}` | "
            f"`{profile['speedup_vs_one_token']:.4f}x` | `{profile['adaptive_step_legal']}` |"
        )
    lines.extend(["", "## Readiness", ""])
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate is a symbolic accepted-token denominator optimizer; it does not execute speculative decode.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_prefix_transaction_ledger_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    layout = result["layout"]
    lines = [
        "# T020 Prefix Transaction Ledger Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| average mutation cost | `{baseline['average_eager_mutation_cost']:.6f}` | "
            f"`{candidate['average_transaction_ledger_cost']:.6f}` | symbolic accept lengths |"
        ),
        (
            f"| deleted mutation cost | `0.000000` | "
            f"`{candidate['average_deleted_cost']:.6f}` | symbolic accept lengths |"
        ),
        (
            f"| speedup vs eager mutation | `1.0000x` | "
            f"`{candidate['speedup_vs_eager_mutation']:.4f}x` | modeled transaction ledger |"
        ),
        (
            f"| rollback copy or compaction required | "
            f"`{baseline['rollback_requires_copy_or_compaction']}` | "
            f"`{candidate['rollback_requires_copy_or_compaction']}` | visibility ledger |"
        ),
        "",
        "## Layout",
        "",
        f"- layers: `{layout['layers']}`",
        f"- heads: `{layout['heads']}`",
        f"- branch length: `{layout['branch_length']}`",
        f"- committed tokens: `{layout['committed_tokens']}`",
        f"- delta entries: `{layout['delta_entries']}`",
        f"- cases enumerated: `{len(layout['cases'])}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate proves symbolic prefix visibility and no-copy rollback only; it does not execute a decoder.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_acceptance_weighted_kv_materialization_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    domain = result["domain"]
    lines = [
        "# T020 Acceptance-Weighted KV Materialization Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| materialization cost | `{baseline['eager_materialization_cost']:.6f}` | "
            f"`{candidate['delayed_materialization_cost']:.6f}` | accepted-span probability |"
        ),
        f"| deleted cost | `0.000000` | `{candidate['deleted_cost']:.6f}` | accepted-span probability |",
        (
            f"| speedup vs eager materialization | `1.0000x` | "
            f"`{candidate['speedup_vs_eager_materialization']:.4f}x` | symbolic materialization gate |"
        ),
        (
            f"| materializes rejected suffix | `{baseline['materializes_rejected_suffix']}` | "
            f"`{candidate['materializes_rejected_suffix']}` | delayed KV materialization |"
        ),
        "",
        "## Domain",
        "",
        f"- branch length: `{domain['branch_length']}`",
        f"- layer groups: `{domain['layer_groups']}`",
        f"- accepted-span probability: `{domain['accepted_span_probability']}`",
        f"- break-even acceptance probability: `{domain['break_even_acceptance_probability']}`",
        "",
        "## Curve",
        "",
        "| p | expected survivors | delayed cost | speedup | beats eager |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in result["curve"]:
        lines.append(
            f"| `{row['acceptance_probability']:.2f}` | `{row['expected_surviving_tokens']:.6f}` | "
            f"`{row['delayed_materialization_cost']:.6f}` | "
            f"`{row['speedup_vs_eager_materialization']:.4f}x` | `{row['delayed_beats_eager']}` |"
        )
    lines.extend(["", "## Readiness", ""])
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate prices delayed KV materialization only; it does not execute a decoder.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_basis_shared_kv_transform_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    proof = result["proof"]
    lines = [
        "# T020 Basis-Shared KV Transform Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| representation crossing cost | `{baseline['representation_crossing_cost']:.6f}` | "
            f"`{candidate['basis_shared_cost']:.6f}` | diagonal dual basis |"
        ),
        f"| deleted crossing cost | `0.000000` | `{candidate['deleted_crossing_cost']:.6f}` | diagonal dual basis |",
        (
            f"| speedup vs representation crossing | `1.0000x` | "
            f"`{candidate['speedup_vs_representation_crossing']:.4f}x` | symbolic basis gate |"
        ),
        (
            f"| requires canonical crossing | `{baseline['requires_canonical_crossing']}` | "
            f"`{candidate['requires_canonical_crossing']}` | dual-basis KV |"
        ),
        "",
        "## Proof",
        "",
        f"- transform class: `{proof['transform_class']}`",
        f"- max absolute dot error: `{proof['max_abs_error']}`",
        f"- condition number: `{proof['condition_number']}`",
        f"- numerical debt: `{proof['numerical_debt']}`",
        f"- attention margin: `{proof['attention_margin']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate proves diagonal dual-basis algebra only; it does not change runtime KV storage.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_ring_exact_tail_receipts_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    baseline = result["baseline"]
    candidate = result["candidate"]
    readiness = result["readiness"]
    lines = [
        "# T020 Ring-Exact Tail Receipts Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Baseline Versus Candidate",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        f"| active context cost | `{baseline['baseline_active_cost']:.6f}` | `{candidate['candidate_ring_tail_cost']:.6f}` | symbolic context policy |",
        f"| deleted active context cost | `0.000000` | `{candidate['deleted_active_context_cost']:.6f}` | symbolic context policy |",
        f"| speedup vs full active context | `1.0000x` | `{candidate['speedup_vs_full_active_context']:.4f}x` | symbolic context policy |",
        f"| GPU KV GiB | `{baseline['baseline_kv_gib']:.6f}` | `{candidate['ring_kv_gib']:.6f}` | exact ring only |",
        f"| exact ring tokens | `{baseline['full_context_tokens']}` | `{candidate['exact_ring_tokens']}` | ring-tail partition |",
        "",
        "## Tail Patterns",
        "",
        "| pattern | reactivation | error bound | margin | declared debt | safe without debt |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in result["patterns"]:
        lines.append(
            f"| `{row['pattern']}` | `{row['tail_reactivation_rate']:.2f}` | "
            f"`{row['tail_attention_error_bound']:.6f}` | `{row['task_margin']:.6f}` | "
            f"`{row['declared_debt_required']}` | `{row['safe_without_debt']}` |"
        )
    lines.extend(["", "## Readiness", ""])
    for key, value in readiness.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate prices a symbolic ring-tail context policy only; it does not run long-context decode.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t020_cumulative_execution_shape_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    composition = result["composition"]
    lines = [
        "# T020 Cumulative Execution-Shape Gate",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Declared Cumulative Speedup",
        "",
        "| item | before | after | status |",
        "|---|---:|---:|---|",
        (
            f"| tokens/s proxy | `{composition['baseline_tokens_per_second']:.6f}` | "
            f"`{composition['candidate_tokens_per_second_proxy']:.6f}` | cumulative declared proxy |"
        ),
        f"| cumulative speedup | `1.0000x` | `{composition['cumulative_speedup']:.4f}x` | composed T020 stack |",
        "",
        "## Numeric Components",
        "",
        "| finding | label | scope | before | after | speedup |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in result["numeric_items"]:
        lines.append(
            f"| `{item['finding']}` | `{item['label']}` | `{item['scope']}` | "
            f"`{item['before']:.6f}` | `{item['after']:.6f}` | `{item['speedup']:.4f}x` |"
        )
    lines.extend(["", "## Structural Components", ""])
    for item in result["structural_items"]:
        lines.append(
            f"- `{item['finding']}` / `{item['label']}`: deleted "
            f"`{item['deleted_dense_mib']:.6f}` MiB and fused "
            f"`{item['fused_residual_adds']}` residual adds in scope `{item['scope']}`."
        )
    lines.extend(["", "## Readiness", ""])
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Component Readiness", ""])
    for key, value in result["component_ready"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    blockers = result["blockers"] or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "This gate declares the implemented T020 findings as one cumulative research stack. "
            "It preserves per-finding before/after evidence and uses the cumulative proxy until "
            "the production decoder path can provide measured tok/s.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t039b_a01_non_row_exact_object_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T039B A01 Non-Row Exact Object Extraction Ledger",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Best Object Family", ""])
    best = result["best_object_family_row"]
    if best is None:
        lines.append("- none")
    else:
        for key in (
            "object_family",
            "sample_or_bundle_id",
            "component_speedup_vs_current_component",
            "covered_body_fraction",
            "charged_atom_or_program_work",
            "dense_work_units",
            "readiness_blockers",
        ):
            lines.append(f"- `{key}`: `{best.get(key)}`")
    lines.extend(["", "## Ledger Summary", ""])
    ledger = result["ledger"]
    for key, value in ledger["coverage_model"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Audited Blocks", ""])
    lines.extend(
        [
            "| tensor | shape | row quotient speedup | best non-row family | best non-row speedup |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for block in result["audited_blocks"]:
        rows = block["block_shape"][0]
        cols = block["block_shape"][1]
        best_block_row = max(
            block["non_row_object_rows"],
            key=lambda row: row["component_speedup_vs_current_component"],
            default=None,
        )
        lines.append(
            f"| `{block['tensor']}` | `{rows}x{cols}` | "
            f"`{block['row_quotient']['row_quotient_speedup']}` | "
            f"`{best_block_row['object_family'] if best_block_row else 'none'}` | "
            f"`{best_block_row['component_speedup_vs_current_component'] if best_block_row else 0.0}` |"
        )
    lines.extend(["", "## Object Family Rows", ""])
    lines.extend(
        [
            "| family | sample | speedup | blockers |",
            "|---|---|---:|---|",
        ]
    )
    for row in result["object_family_rows"]:
        lines.append(
            f"| `{row['object_family']}` | `{row['sample_or_bundle_id']}` | "
            f"`{row['component_speedup_vs_current_component']}` | "
            f"`{row['readiness_blockers']}` |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_t039b_a02_gauge_sat_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "# T039B A02 Gauge SAT Canonical Tail Ledger",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Readiness",
        "",
    ]
    for key, value in result["readiness"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gauge SAT", ""])
    for key, value in result["gauge_sat"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Regions", ""])
    lines.extend(
        [
            "| region | fraction | forced canonical | charged speedup | blockers |",
            "|---|---:|---|---:|---|",
        ]
    )
    for row in result["regions"]:
        lines.append(
            f"| `{row['name']}` | `{row['body_fraction']}` | "
            f"`{row['forced_canonical']}` | `{row['charged_speedup']}` | "
            f"`{row['readiness_blockers']}` |"
        )
    lines.extend(["", "## Ledger Summary", ""])
    ledger = result["ledger"]
    for key, value in ledger["coverage_model"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in result["blockers"] or ["none"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"] or ["none"]:
        lines.append(f"- {action}")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t002-kv-margin-sweep", action="store_true")
    parser.add_argument("--t003-receipt-kv-gate", action="store_true")
    parser.add_argument("--t004-real-qwen-bridge", action="store_true")
    parser.add_argument("--t005-real-weight-sentinel", action="store_true")
    parser.add_argument("--t006-real-residency-block-smoke", action="store_true")
    parser.add_argument("--t007-real-layer-smoke", action="store_true")
    parser.add_argument("--t008-compressed-generation-smoke", action="store_true")
    parser.add_argument("--t009-t013-runtime-scaler", action="store_true")
    parser.add_argument("--t014-denominator-tightening", action="store_true")
    parser.add_argument("--t015-packed-qwen-inference", action="store_true")
    parser.add_argument("--t016-persistent-packed-cache", action="store_true")
    parser.add_argument("--t017-packed-linear-kernel", action="store_true")
    parser.add_argument("--t017-kernel-qwen-inference", action="store_true")
    parser.add_argument("--t020-cumulative-execution-shape", action="store_true")
    parser.add_argument("--t031-macro-architecture-stack", action="store_true")
    parser.add_argument("--t034-gpu-real-runtime-gates", action="store_true")
    parser.add_argument("--t018-resident-payload-qwen-inference", action="store_true")
    parser.add_argument("--t019-binary-payload-qwen-inference", action="store_true")
    parser.add_argument("--t020-acceptance-weighted-kv-materialization", action="store_true")
    parser.add_argument("--t020-affine-packed-view", action="store_true")
    parser.add_argument("--t020-basis-shared-kv-transform", action="store_true")
    parser.add_argument("--t020-conversion-time-arena-compiler", action="store_true")
    parser.add_argument("--t020-device-credit-flow", action="store_true")
    parser.add_argument("--t020-expected-accepted-span", action="store_true")
    parser.add_argument("--t020-execution-order-slab", action="store_true")
    parser.add_argument("--t020-immutable-arena-receipt", action="store_true")
    parser.add_argument("--t020-packed-work-coalescing", action="store_true")
    parser.add_argument("--t020-prefix-transaction-ledger", action="store_true")
    parser.add_argument("--t020-recompute-on-reject", action="store_true")
    parser.add_argument("--t020-ring-exact-tail-receipts", action="store_true")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3-32B")
    parser.add_argument("--local-model-dir", type=Path, default=None)
    parser.add_argument("--no-hf-config-fetch", action="store_true")
    parser.add_argument("--sentinel-sample-rows", type=int, default=8)
    parser.add_argument("--sentinel-sample-cols", type=int, default=8)
    parser.add_argument("--sentinel-min-required-tensors", type=int, default=4)
    parser.add_argument("--block-smoke-rows", type=int, default=64)
    parser.add_argument("--block-smoke-cols", type=int, default=256)
    parser.add_argument("--block-smoke-batch", type=int, default=16)
    parser.add_argument("--block-smoke-min-required", type=int, default=4)
    parser.add_argument("--layer-smoke-index", type=int, default=0)
    parser.add_argument("--layer-smoke-hidden-cols", type=int, default=256)
    parser.add_argument("--layer-smoke-attention-width", type=int, default=128)
    parser.add_argument("--layer-smoke-mlp-width", type=int, default=384)
    parser.add_argument("--layer-smoke-sequence-tokens", type=int, default=8)
    parser.add_argument("--generation-layer-indices", type=_parse_int_csv, default=(0, 1))
    parser.add_argument("--generation-hidden-cols", type=int, default=256)
    parser.add_argument("--generation-attention-width", type=int, default=128)
    parser.add_argument("--generation-mlp-width", type=int, default=384)
    parser.add_argument("--generation-vocab-rows", type=int, default=512)
    parser.add_argument("--generation-prompt-tokens", type=_parse_int_csv, default=(1, 2, 3, 4))
    parser.add_argument("--generation-max-new-tokens", type=int, default=4)
    parser.add_argument("--runtime-full-layer-index", type=int, default=0)
    parser.add_argument("--runtime-scale-layer-indices", type=_parse_int_csv, default=(0, 1, 2, 3))
    parser.add_argument("--runtime-scale-row-limit", type=int, default=512)
    parser.add_argument("--runtime-projection-batch", type=int, default=1)
    parser.add_argument("--runtime-prompt-text", default="Local compressed Qwen smoke:")
    parser.add_argument("--runtime-allow-tokenizer-fallback", action="store_true")
    parser.add_argument("--runtime-kv-context-tokens", type=_parse_int_csv, default=(4096, 8192, 16384))
    parser.add_argument("--runtime-kv-effective-bits", type=float, default=4.0)
    parser.add_argument("--runtime-exact-recent-tokens", type=int, default=1024)
    parser.add_argument("--runtime-bandwidth-gib-per-second", type=float, default=514.0)
    parser.add_argument("--runtime-scaler-overhead-gib", type=float, default=0.55)
    parser.add_argument("--t014-mlp-layer-index", type=int, default=0)
    parser.add_argument("--t014-mlp-row-limit", type=int, default=0)
    parser.add_argument("--t014-projection-batch", type=int, default=1)
    parser.add_argument("--t014-target-mlp-bits", type=float, default=2.45)
    parser.add_argument("--t014-context-tokens", type=int, default=16384)
    parser.add_argument("--t014-bandwidth-gib-per-second", type=float, default=514.0)
    parser.add_argument("--t014-no-stretch", action="store_true")
    parser.add_argument("--t015-prompt-text", default="Local compressed Qwen smoke:")
    parser.add_argument("--t015-max-new-tokens", type=int, default=1)
    parser.add_argument("--t015-layer-limit", type=int, default=0)
    parser.add_argument("--t015-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--t015-target-mlp-bits", type=float, default=2.45)
    parser.add_argument("--t015-lm-head-chunk-rows", type=int, default=2048)
    parser.add_argument("--t016-prompt-text", default="Local compressed Qwen smoke:")
    parser.add_argument("--t016-max-new-tokens", type=int, default=1)
    parser.add_argument("--t016-layer-limit", type=int, default=0)
    parser.add_argument("--t016-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--t016-target-mlp-bits", type=float, default=2.45)
    parser.add_argument("--t016-lm-head-chunk-rows", type=int, default=2048)
    parser.add_argument("--t016-cache-dir", type=Path, default=None)
    parser.add_argument("--t016-refresh-cache", action="store_true")
    parser.add_argument("--t017-rows", type=int, default=512)
    parser.add_argument("--t017-cols", type=int, default=512)
    parser.add_argument("--t017-tokens", type=int, default=4)
    parser.add_argument("--t017-repair-fraction", type=float, default=0.03)
    parser.add_argument("--t017-iterations", type=int, default=25)
    parser.add_argument("--t017-seed", type=int, default=17017)
    parser.add_argument("--t017-prompt-text", default="Local compressed Qwen smoke:")
    parser.add_argument("--t017-max-new-tokens", type=int, default=1)
    parser.add_argument("--t017-layer-limit", type=int, default=1)
    parser.add_argument("--t017-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--t017-target-mlp-bits", type=float, default=2.45)
    parser.add_argument("--t017-lm-head-chunk-rows", type=int, default=2048)
    parser.add_argument("--t017-cache-dir", type=Path, default=None)
    parser.add_argument("--t017-refresh-cache", action="store_true")
    parser.add_argument("--t017-device-credit-flow", action="store_true")
    parser.add_argument("--t017-packed-work-coalescing", action="store_true")
    parser.add_argument("--t018-no-preload-payloads", action="store_true")
    parser.add_argument("--t020-baseline-total-seconds", type=float, default=35.6344)
    parser.add_argument("--t020-baseline-payload-load-seconds", type=float, default=28.0466)
    parser.add_argument("--t020-baseline-kernel-seconds", type=float, default=5.6611)
    parser.add_argument("--t020-baseline-tokens-per-second", type=float, default=0.028063)
    parser.add_argument("--t020-device-budget-gib", type=float, default=12.0)
    parser.add_argument("--t020-baseline-denominator-gib", type=float, default=11.8424)
    parser.add_argument("--t020-staged-payload-window-mib", type=float, default=48.0)
    parser.add_argument("--t020-accumulator-window-mib", type=float, default=64.0)
    parser.add_argument("--t020-credit-state-mib", type=float, default=8.0)
    parser.add_argument("--t020-sync-workspace-mib", type=float, default=8.0)
    parser.add_argument("--t020-decode-seconds", type=float, default=0.0)
    parser.add_argument("--t020-epilogue-seconds", type=float, default=0.0)
    parser.add_argument("--t020-credit-overhead-fraction", type=float, default=0.02)
    parser.add_argument("--t020-sync-penalty-seconds", type=float, default=0.0)
    parser.add_argument("--t020-min-speedup", type=float, default=1.10)
    parser.add_argument("--t020-coalescing-layers", type=int, default=64)
    parser.add_argument("--t020-coalescing-hidden-size", type=int, default=5120)
    parser.add_argument("--t020-coalescing-tokens", type=int, default=1)
    parser.add_argument("--t020-coalescing-activation-bytes", type=int, default=4)
    parser.add_argument("--t020-coalescing-min-legal-outputs", type=int, default=2)
    parser.add_argument("--t020-slab-baseline-objects", type=int, default=82)
    parser.add_argument("--t020-slab-baseline-acquisition-seconds", type=float, default=0.5214)
    parser.add_argument("--t020-slab-fixed-object-fraction", type=float, default=0.70)
    parser.add_argument("--t020-slab-offset-walk-nanoseconds", type=float, default=50.0)
    parser.add_argument("--t020-slab-min-acquisition-speedup", type=float, default=2.0)
    parser.add_argument("--t020-slab-lm-head-rows", type=int, default=151936)
    parser.add_argument("--t020-slab-lm-head-chunk-rows", type=int, default=2048)
    parser.add_argument("--t020-receipt-baseline-dynamic-discovery-seconds", type=float, default=0.5214)
    parser.add_argument("--t020-receipt-expected-tokens-between-rebuilds", type=int, default=50000)
    parser.add_argument("--t020-receipt-offline-build-seconds", type=float, default=90.0)
    parser.add_argument("--t020-receipt-open-arena-seconds", type=float, default=0.002)
    parser.add_argument("--t020-receipt-descriptor-verify-microseconds", type=float, default=12.0)
    parser.add_argument("--t020-receipt-root-lookup-seconds", type=float, default=0.001)
    parser.add_argument("--t020-receipt-min-speedup", type=float, default=1.10)
    parser.add_argument("--t020-receipt-lm-head-rows", type=int, default=151936)
    parser.add_argument("--t020-receipt-lm-head-chunk-rows", type=int, default=2048)
    parser.add_argument("--t020-compiler-baseline-visible-denominator-seconds", type=float, default=0.5214)
    parser.add_argument("--t020-compiler-receipt-admission-seconds", type=float, default=0.005784)
    parser.add_argument("--t020-compiler-subset-descriptors", type=int, default=8)
    parser.add_argument("--t020-compiler-min-saved-fraction", type=float, default=0.10)
    parser.add_argument("--t020-affine-rows", type=int, default=16)
    parser.add_argument("--t020-affine-cols", type=int, default=32)
    parser.add_argument("--t020-affine-group-cols", type=int, default=8)
    parser.add_argument("--t020-affine-visible-denominator-seconds", type=float, default=0.5214)
    parser.add_argument("--t020-affine-materialization-cost-removed-seconds", type=float, default=0.1800)
    parser.add_argument("--t020-affine-decode-overhead-seconds", type=float, default=0.0180)
    parser.add_argument("--t020-affine-coalescing-penalty-seconds", type=float, default=0.0120)
    parser.add_argument("--t020-affine-min-saved-fraction", type=float, default=0.10)
    parser.add_argument("--t020-recompute-branch-depth", type=int, default=4)
    parser.add_argument("--t020-recompute-branch-width", type=int, default=4)
    parser.add_argument("--t020-recompute-base-acceptance", type=float, default=0.60)
    parser.add_argument("--t020-recompute-kv-write-cost", type=float, default=1.0)
    parser.add_argument("--t020-recompute-memory-pressure-cost", type=float, default=0.25)
    parser.add_argument("--t020-recompute-depth-memory-pressure-step", type=float, default=0.05)
    parser.add_argument("--t020-recompute-cost", type=float, default=2.20)
    parser.add_argument("--t020-recompute-cleanup-cost", type=float, default=0.20)
    parser.add_argument("--t020-recompute-receipt-cost", type=float, default=0.05)
    parser.add_argument("--t020-recompute-max-store-fraction", type=float, default=0.80)
    parser.add_argument("--t020-recompute-min-speedup", type=float, default=1.10)
    parser.add_argument("--t020-span-risk-margin", type=float, default=0.03)
    parser.add_argument("--t020-span-min-speedup", type=float, default=1.10)
    parser.add_argument("--t020-span-base-cost", type=float, default=0.36)
    parser.add_argument("--t020-span-verify-cost-per-token", type=float, default=0.34)
    parser.add_argument("--t020-span-delayed-kv-cost-per-token", type=float, default=0.05)
    parser.add_argument("--t020-span-lm-head-cost-per-token", type=float, default=0.06)
    parser.add_argument("--t020-span-rollback-cost-per-rejected-token", type=float, default=0.08)
    parser.add_argument("--t020-span-schedule-gap-cost", type=float, default=0.10)
    parser.add_argument("--t020-span-graph-specialization-cost", type=float, default=0.02)
    parser.add_argument("--t020-prefix-layers", type=int, default=2)
    parser.add_argument("--t020-prefix-heads", type=int, default=2)
    parser.add_argument("--t020-prefix-branch-length", type=int, default=4)
    parser.add_argument("--t020-prefix-committed-tokens", type=int, default=3)
    parser.add_argument("--t020-prefix-min-speedup", type=float, default=1.10)
    parser.add_argument("--t020-materialization-accepted-span-probability", type=float, default=0.92)
    parser.add_argument("--t020-materialization-recompute-to-kv-ratio", type=float, default=0.04)
    parser.add_argument("--t020-materialization-receipt-cost-per-token-layer", type=float, default=0.0125)
    parser.add_argument("--t020-materialization-conditional-schedule-cost", type=float, default=0.10)
    parser.add_argument("--t020-materialization-min-speedup", type=float, default=1.10)
    parser.add_argument("--t020-basis-vector-size", type=int, default=16)
    parser.add_argument("--t020-basis-epsilon-numeric", type=float, default=1e-6)
    parser.add_argument("--t020-basis-attention-margin", type=float, default=1e-3)
    parser.add_argument("--t020-basis-min-speedup", type=float, default=1.10)
    parser.add_argument("--t020-ring-full-context-tokens", type=int, default=16384)
    parser.add_argument("--t020-ring-min-exact-tokens", type=int, default=2048)
    parser.add_argument("--t020-ring-min-speedup", type=float, default=1.10)
    parser.add_argument("--t021-runtime-obligation-measure", action="store_true")
    parser.add_argument("--t021-transaction-receipt-gate", action="store_true")
    parser.add_argument("--t021-page-activation-certificate", action="store_true")
    parser.add_argument("--t021-cumulative-decoder-runtime-path", action="store_true")
    parser.add_argument("--t021-qwen-runtime", action="store_true")
    parser.add_argument("--t022-qwen-runtime", action="store_true")
    parser.add_argument("--t023-qwen-runtime", action="store_true")
    parser.add_argument("--t024-qwen-runtime", action="store_true")
    parser.add_argument("--t025-qwen-runtime", action="store_true")
    parser.add_argument("--t026-qwen-runtime", action="store_true")
    parser.add_argument("--t027-qwen-runtime", action="store_true")
    parser.add_argument("--t028-native-generation-speed", action="store_true")
    parser.add_argument("--t029-speed-gpu-efficiency", action="store_true")
    parser.add_argument("--t030-speed-gpu-efficiency", action="store_true")
    parser.add_argument("--t035-runtime-guardrail-gate", action="store_true")
    parser.add_argument("--t035-row-successor-gate", action="store_true")
    parser.add_argument("--t035-row-successor-runtime-path", action="store_true")
    parser.add_argument("--t036-a05-block-verifier-gate", action="store_true")
    parser.add_argument("--t022-device-payload-budget-gib", type=float, default=9.5)
    parser.add_argument("--t023-device-payload-budget-gib", type=float, default=0.0)
    parser.add_argument("--t025-device-payload-budget-gib", type=float, default=10.5)
    parser.add_argument("--t026-device-payload-budget-gib", type=float, default=10.5)
    parser.add_argument("--t027-device-payload-budget-gib", type=float, default=10.5)
    parser.add_argument("--t028-device-payload-budget-gib", type=float, default=10.5)
    parser.add_argument("--t029-device-payload-budget-gib", type=float, default=10.5)
    parser.add_argument("--t030-device-payload-budget-gib", type=float, default=10.5)
    parser.add_argument("--t026-packed-linear-block-m", type=int, default=16)
    parser.add_argument("--t026-packed-linear-block-c", type=int, default=128)
    parser.add_argument("--t026-packed-linear-block-r", type=int, default=32)
    parser.add_argument("--t028-packed-linear-block-m", type=int, default=8)
    parser.add_argument("--t028-packed-linear-block-c", type=int, default=128)
    parser.add_argument("--t028-packed-linear-block-r", type=int, default=32)
    parser.add_argument("--t029-packed-linear-block-m", type=int, default=8)
    parser.add_argument("--t029-packed-linear-block-c", type=int, default=128)
    parser.add_argument("--t029-packed-linear-block-r", type=int, default=32)
    parser.add_argument("--t030-packed-linear-block-m", type=int, default=8)
    parser.add_argument("--t030-packed-linear-block-c", type=int, default=128)
    parser.add_argument("--t030-packed-linear-block-r", type=int, default=32)
    parser.add_argument("--t030-enable-fused-qkv-kernel", action="store_true")
    parser.add_argument("--t026-fused-mlp-gate-up-kernel", action="store_true")
    parser.add_argument("--t035-row-successor-buffer-gib", type=float, default=0.03125)
    parser.add_argument("--t035-verifier-buffer-gib", type=float, default=0.03125)
    parser.add_argument("--t035-graph-memory-node-gib", type=float, default=0.046875)
    parser.add_argument("--t035-descriptor-tape-gib", type=float, default=0.015625)
    parser.add_argument("--t035-compact-fragment-gib", type=float, default=0.0625)
    parser.add_argument("--t035-token-commit-buffer-gib", type=float, default=0.015625)
    parser.add_argument("--t035-hidden-dense-gib", type=float, default=0.0)
    parser.add_argument("--t035-reserved-pool-growth-gib", type=float, default=0.0)
    parser.add_argument("--t035-exact-splice-peak-budget-gib", type=float, default=0.25)
    parser.add_argument("--t035-row-successor-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--t035-row-successor-vocab-rows", type=int, default=4096)
    parser.add_argument("--t035-row-successor-hidden-size", type=int, default=512)
    parser.add_argument("--t035-row-successor-chunk-rows", type=int, default=512)
    parser.add_argument("--t035-row-successor-seed", type=int, default=3503)
    parser.add_argument("--t036-block-sizes", default="16,32,64")
    parser.add_argument(
        "--t036-verifier-mode",
        choices=("serial_full64_replay", "projected_sublinear"),
        default="serial_full64_replay",
    )
    parser.add_argument("--t036-extra-seconds", type=float, default=0.0)
    parser.add_argument("--t036-shadow-kv-gib", type=float, default=0.0)
    parser.add_argument("--t036-scratch-gib", type=float, default=0.0)
    parser.add_argument("--t036-graph-memory-gib", type=float, default=0.0)
    parser.add_argument("--t036-dense-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t036-reserved-pool-growth-gib", type=float, default=0.0)
    parser.add_argument("--t036-device-budget-gib", type=float, default=12.0)
    parser.add_argument("--t036-break-state-equality", action="store_true")
    parser.add_argument("--t036-break-token-equality", action="store_true")
    parser.add_argument("--t037-exact-suffix-prefill-gate", action="store_true")
    parser.add_argument("--t037-block-sizes", default="16,32,64")
    parser.add_argument("--t037-prefix-tokens", type=int, default=8)
    parser.add_argument("--t037-vocab-size", type=int, default=257)
    parser.add_argument("--t037-full64-suffix-prefill-seconds", type=float, default=None)
    parser.add_argument("--t037-extra-seconds", type=float, default=0.0)
    parser.add_argument("--t037-shadow-kv-gib", type=float, default=0.0)
    parser.add_argument("--t037-scratch-gib", type=float, default=0.0)
    parser.add_argument("--t037-graph-memory-gib", type=float, default=0.0)
    parser.add_argument("--t037-dense-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t037-device-budget-gib", type=float, default=12.0)
    parser.add_argument("--t037-timing-source", default="missing_full64_measurement")
    parser.add_argument("--t037-direct-denominator-gate", action="store_true")
    parser.add_argument("--t037-direct-seconds-per-token", type=float, default=None)
    parser.add_argument("--t037-direct-kernel-seconds-per-token", type=float, default=None)
    parser.add_argument("--t037-direct-non-kernel-seconds-per-token", type=float, default=None)
    parser.add_argument("--t037-direct-peak-cuda-gib", type=float, default=10.652063369750977)
    parser.add_argument("--t037-direct-hidden-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t037-direct-setup-seconds-per-token", type=float, default=0.0)
    parser.add_argument("--t037-direct-break-exactness", action="store_true")
    parser.add_argument("--t038-kernel-body-roofline-gate", action="store_true")
    parser.add_argument("--t038-resident-payload-gib", type=float, default=10.458633184432983)
    parser.add_argument("--t038-sustained-bandwidth-gib-s", type=float, default=514.0)
    parser.add_argument("--t038-hot-read-multiplier", type=float, default=1.0)
    parser.add_argument("--t038-arithmetic-seconds-per-token", type=float, default=0.0)
    parser.add_argument("--t038-control-seconds-per-token", type=float, default=0.0)
    parser.add_argument("--t038-setup-seconds-per-token", type=float, default=0.0)
    parser.add_argument("--t038-peak-cuda-gib", type=float, default=10.652063369750977)
    parser.add_argument("--t038-hidden-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t038-decoded-dense-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t038-replacement-layout-gib", type=float, default=None)
    parser.add_argument("--t038-compact-affine-map-fraction", type=float, default=1.0)
    parser.add_argument("--t038-sign-plane-map-fraction", type=float, default=0.0)
    parser.add_argument("--t038-dense-map-fraction", type=float, default=0.0)
    parser.add_argument("--t038-break-exact-storage-semantics", action="store_true")
    parser.add_argument("--t038-representative-map-gate", action="store_true")
    parser.add_argument("--t038-representative-speedup", type=float, default=4.7723)
    parser.add_argument("--t038-representative-coverage", type=float, default=0.80)
    parser.add_argument("--t038-representative-hidden-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t038-representative-peak-cuda-gib", type=float, default=10.652063369750977)
    parser.add_argument("--t038-representative-break-exactness", action="store_true")
    parser.add_argument("--t038b-coverage-denominator-gate", action="store_true")
    parser.add_argument("--t038b-component-names", type=_parse_str_csv, default=("exact_compact_body", "unchanged_leftover_body"))
    parser.add_argument("--t038b-component-fractions", type=_parse_float_csv, default=(0.80, 0.20))
    parser.add_argument("--t038b-component-speedups", type=_parse_float_csv, default=(7.573753504548222, 1.0))
    parser.add_argument("--t038b-minimum-accelerated-body-fraction", type=float, default=0.80)
    parser.add_argument("--t038b-hidden-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t038b-decoded-dense-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t038b-residual-fraction", type=float, default=0.0)
    parser.add_argument("--t038b-break-exact-arithmetic", action="store_true")
    parser.add_argument("--t038b-break-exact-layout-semantics", action="store_true")
    parser.add_argument("--t038b-break-layout-replaces-storage", action="store_true")
    parser.add_argument("--t038b-peak-cuda-gib", type=float, default=10.652063369750977)
    parser.add_argument("--t039-a01-exact-map-object-ledger-gate", action="store_true")
    parser.add_argument("--t039-a01-component-names", type=_parse_str_csv, default=("unextracted_current_body",))
    parser.add_argument("--t039-a01-component-fractions", type=_parse_float_csv, default=(1.0,))
    parser.add_argument("--t039-a01-component-speedups", type=_parse_float_csv, default=(1.0,))
    parser.add_argument("--t039-a01-object-equivalence-proven", action="store_true")
    parser.add_argument("--t039-a01-representative-real-maps-audited", type=int, default=0)
    parser.add_argument("--t039-a01-residual-fraction", type=float, default=0.0)
    parser.add_argument("--t039-a01-hidden-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t039-a01-decoded-dense-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t039-a01-offload-or-setup-gain-counted", action="store_true")
    parser.add_argument("--t039-a01-layout-replaces-storage", action="store_true")
    parser.add_argument("--t039-a01-break-exact-arithmetic-contract", action="store_true")
    parser.add_argument("--t039-a01-peak-cuda-gib", type=float, default=10.652063369750977)
    parser.add_argument("--t039-a02-semantic-layout-ledger-gate", action="store_true")
    parser.add_argument("--t039-a02-component-names", type=_parse_str_csv, default=("unproven_semantic_layout_body",))
    parser.add_argument("--t039-a02-component-fractions", type=_parse_float_csv, default=(1.0,))
    parser.add_argument("--t039-a02-component-speedups", type=_parse_float_csv, default=(1.0,))
    parser.add_argument("--t039-a02-gauge-contracts-proven", action="store_true")
    parser.add_argument("--t039-a02-semantic-boundaries-priced", action="store_true")
    parser.add_argument("--t039-a02-forced-canonical-fraction", type=float, default=1.0)
    parser.add_argument("--t039-a02-residual-fraction", type=float, default=0.0)
    parser.add_argument("--t039-a02-hidden-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t039-a02-decoded-dense-sidecar-gib", type=float, default=0.0)
    parser.add_argument("--t039-a02-offload-or-setup-gain-counted", action="store_true")
    parser.add_argument("--t039-a02-layout-replaces-storage", action="store_true")
    parser.add_argument("--t039-a02-break-exact-arithmetic-contract", action="store_true")
    parser.add_argument("--t039-a02-peak-cuda-gib", type=float, default=10.652063369750977)
    parser.add_argument("--t039-real-map-object-audit", action="store_true")
    parser.add_argument("--t039b-a01-non-row-exact-object-ledger", action="store_true")
    parser.add_argument("--t039b-a02-gauge-sat-ledger", action="store_true")
    parser.add_argument(
        "--t039-real-map-audit-tensors",
        type=_parse_str_csv,
        default=(
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.63.mlp.down_proj.weight",
        ),
    )
    parser.add_argument("--t039-real-map-audit-rows", type=int, default=64)
    parser.add_argument("--t039-real-map-audit-cols", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--metadata-bytes", type=int, default=256)
    parser.add_argument("--scale-bytes", type=int, default=128)
    parser.add_argument("--correction-bytes", type=int, default=512)
    parser.add_argument("--scratch-bytes", type=int, default=128)
    parser.add_argument("--launch-fragmentation-bytes", type=int, default=0)
    parser.add_argument("--runtime-overhead-gib", type=float, default=1.0)
    parser.add_argument("--mlp-effective-bits", type=float, default=2.65)
    parser.add_argument("--attention-effective-bits", type=float, default=3.50)
    parser.add_argument("--embedding-effective-bits", type=float, default=4.0)
    parser.add_argument("--output-head-effective-bits", type=float, default=4.0)
    parser.add_argument("--synthetic-bandwidth-gib-per-second", type=float, default=500.0)
    parser.add_argument("--weight-streams-per-token", type=float, default=1.0)
    parser.add_argument("--kv-cache-gib-per-token", type=float, default=0.0)
    parser.add_argument("--max-visible-energy-ratio", type=float, default=0.35)
    parser.add_argument("--max-visible-rank-ratio", type=float, default=0.50)
    parser.add_argument("--logit-error-radius", type=float, default=0.08)
    parser.add_argument("--gate-residual-bound", type=float, default=0.35)
    parser.add_argument("--gate-margin", type=float, default=0.05)
    parser.add_argument("--min-gate-accepted-fraction", type=float, default=0.60)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_modes = [
        args.t002_kv_margin_sweep,
        args.t003_receipt_kv_gate,
        args.t004_real_qwen_bridge,
        args.t005_real_weight_sentinel,
        args.t006_real_residency_block_smoke,
        args.t007_real_layer_smoke,
        args.t008_compressed_generation_smoke,
        args.t009_t013_runtime_scaler,
        args.t014_denominator_tightening,
        args.t015_packed_qwen_inference,
        args.t016_persistent_packed_cache,
        args.t017_packed_linear_kernel,
        args.t017_kernel_qwen_inference,
        args.t018_resident_payload_qwen_inference,
        args.t019_binary_payload_qwen_inference,
        args.t020_cumulative_execution_shape,
        args.t020_acceptance_weighted_kv_materialization,
        args.t020_affine_packed_view,
        args.t020_basis_shared_kv_transform,
        args.t020_conversion_time_arena_compiler,
        args.t020_device_credit_flow,
        args.t020_expected_accepted_span,
        args.t020_execution_order_slab,
        args.t020_immutable_arena_receipt,
        args.t020_packed_work_coalescing,
        args.t020_prefix_transaction_ledger,
        args.t020_recompute_on_reject,
        args.t020_ring_exact_tail_receipts,
        args.t021_runtime_obligation_measure,
        args.t021_transaction_receipt_gate,
        args.t021_page_activation_certificate,
        args.t021_cumulative_decoder_runtime_path,
        args.t021_qwen_runtime,
        args.t022_qwen_runtime,
        args.t023_qwen_runtime,
        args.t024_qwen_runtime,
        args.t025_qwen_runtime,
        args.t026_qwen_runtime,
        args.t027_qwen_runtime,
        args.t028_native_generation_speed,
        args.t029_speed_gpu_efficiency,
        args.t030_speed_gpu_efficiency,
        args.t031_macro_architecture_stack,
        args.t034_gpu_real_runtime_gates,
        args.t035_runtime_guardrail_gate,
        args.t035_row_successor_gate,
        args.t035_row_successor_runtime_path,
        args.t036_a05_block_verifier_gate,
        args.t037_exact_suffix_prefill_gate,
        args.t037_direct_denominator_gate,
        args.t038_kernel_body_roofline_gate,
        args.t038_representative_map_gate,
        args.t038b_coverage_denominator_gate,
        args.t039_a01_exact_map_object_ledger_gate,
        args.t039_a02_semantic_layout_ledger_gate,
        args.t039_real_map_object_audit,
        args.t039b_a01_non_row_exact_object_ledger,
        args.t039b_a02_gauge_sat_ledger,
    ]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise SystemExit(
            "--t002-kv-margin-sweep, --t003-receipt-kv-gate, --t004-real-qwen-bridge, "
            "--t005-real-weight-sentinel, --t006-real-residency-block-smoke, and "
            "--t007-real-layer-smoke, --t008-compressed-generation-smoke, and "
            "--t009-t013-runtime-scaler, --t014-denominator-tightening, "
            "--t015-packed-qwen-inference, --t016-persistent-packed-cache, "
            "--t017-packed-linear-kernel, --t017-kernel-qwen-inference, "
            "--t018-resident-payload-qwen-inference, "
            "--t019-binary-payload-qwen-inference, "
            "--t020-cumulative-execution-shape, "
            "--t020-acceptance-weighted-kv-materialization, "
            "--t020-affine-packed-view, --t020-basis-shared-kv-transform, "
            "--t020-conversion-time-arena-compiler, "
            "--t020-device-credit-flow, "
            "--t020-expected-accepted-span, "
            "--t020-execution-order-slab, "
            "--t020-immutable-arena-receipt, --t020-packed-work-coalescing, "
            "--t020-prefix-transaction-ledger, "
            "--t020-recompute-on-reject, --t020-ring-exact-tail-receipts, "
            "--t021-runtime-obligation-measure, --t021-transaction-receipt-gate, "
            "--t021-page-activation-certificate, and "
            "--t021-cumulative-decoder-runtime-path, --t021-qwen-runtime, "
            "--t022-qwen-runtime, --t023-qwen-runtime, --t024-qwen-runtime, "
            "--t025-qwen-runtime, --t026-qwen-runtime, --t027-qwen-runtime, "
            "--t028-native-generation-speed, --t029-speed-gpu-efficiency, "
            "--t030-speed-gpu-efficiency, --t031-macro-architecture-stack, "
            "--t034-gpu-real-runtime-gates, --t035-runtime-guardrail-gate, "
            "--t035-row-successor-gate, --t035-row-successor-runtime-path, "
            "--t036-a05-block-verifier-gate, and "
            "--t037-exact-suffix-prefill-gate, and "
            "--t037-direct-denominator-gate, and "
            "--t038-kernel-body-roofline-gate, and "
            "--t038-representative-map-gate, and "
            "--t038b-coverage-denominator-gate, and "
            "--t039-a01-exact-map-object-ledger-gate, and "
            "--t039-a02-semantic-layout-ledger-gate, and "
            "--t039-real-map-object-audit, and "
            "--t039b-a01-non-row-exact-object-ledger, and "
            "--t039b-a02-gauge-sat-ledger "
            "are mutually exclusive"
        )
    if args.t002_kv_margin_sweep:
        result = run_t002_kv_margin_sweep(
            bandwidth_gib_per_second=args.synthetic_bandwidth_gib_per_second,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t002-kv-margin-{stamp}")
        write_t002_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "overall": result["overall"],
                    "candidate_summary": result["candidate_summary"],
                    "validations": result["validations"],
                },
                indent=2,
            )
        )
        return
    if args.t003_receipt_kv_gate:
        result = run_t003_receipt_kv_gate(
            bandwidth_gib_per_second=args.synthetic_bandwidth_gib_per_second,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t003-receipt-kv-{stamp}")
        write_t003_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "overall": result["overall"],
                    "validations": result["validations"],
                },
                indent=2,
            )
        )
        return
    if args.t004_real_qwen_bridge:
        result = run_t004_real_qwen_bridge(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t004-real-qwen-bridge-{stamp}")
        write_t004_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t034_gpu_real_runtime_gates:
        result = run_t034_gpu_real_runtime_gates(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t030_device_payload_budget_gib,
            packed_linear_block_m=args.t030_packed_linear_block_m,
            packed_linear_block_c=args.t030_packed_linear_block_c,
            packed_linear_block_r=args.t030_packed_linear_block_r,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t034-gpu-real-runtime-gates-{stamp}")
        write_t034_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "comparison": result["t034_comparison"],
                    "gates": result["t034_gates"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t035_runtime_guardrail_gate:
        result = run_t035_runtime_guardrail_gate(
            row_successor_buffer_gib=args.t035_row_successor_buffer_gib,
            verifier_buffer_gib=args.t035_verifier_buffer_gib,
            graph_memory_node_gib=args.t035_graph_memory_node_gib,
            descriptor_tape_gib=args.t035_descriptor_tape_gib,
            compact_fragment_gib=args.t035_compact_fragment_gib,
            token_commit_buffer_gib=args.t035_token_commit_buffer_gib,
            hidden_dense_gib=args.t035_hidden_dense_gib,
            reserved_pool_growth_gib=args.t035_reserved_pool_growth_gib,
            exact_splice_peak_budget_gib=args.t035_exact_splice_peak_budget_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t035-runtime-guardrail-gate-{stamp}")
        write_t035_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "scratch_contract": result["scratch_contract"],
                    "projection": result["projection"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t035_row_successor_gate:
        result = run_t035_row_successor_gate(
            device=args.t035_row_successor_device,
            vocab_rows=args.t035_row_successor_vocab_rows,
            hidden_size=args.t035_row_successor_hidden_size,
            chunk_rows=args.t035_row_successor_chunk_rows,
            seed=args.t035_row_successor_seed,
            exact_splice_peak_budget_gib=args.t035_exact_splice_peak_budget_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t035-row-successor-gate-{stamp}")
        write_t035_row_successor_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "candidate": result["candidate"],
                    "allocation_census": result["allocation_census"],
                    "projection": result["projection"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t035_row_successor_runtime_path:
        result = run_t035_row_successor_runtime_path(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
            device_payload_budget_gib=args.t030_device_payload_budget_gib,
            packed_linear_block_m=args.t030_packed_linear_block_m,
            packed_linear_block_c=args.t030_packed_linear_block_c,
            packed_linear_block_r=args.t030_packed_linear_block_r,
            exact_splice_peak_budget_gib=args.t035_exact_splice_peak_budget_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t035-row-successor-runtime-path-{stamp}")
        write_t035_row_successor_runtime_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "comparison": result["comparison"],
                    "projection": result["projection"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t036_a05_block_verifier_gate:
        result = run_t036_a05_block_verifier_gate(
            block_sizes=args.t036_block_sizes,
            verifier_mode=args.t036_verifier_mode,
            extra_seconds=args.t036_extra_seconds,
            shadow_kv_gib=args.t036_shadow_kv_gib,
            scratch_gib=args.t036_scratch_gib,
            graph_memory_gib=args.t036_graph_memory_gib,
            dense_sidecar_gib=args.t036_dense_sidecar_gib,
            reserved_pool_growth_gib=args.t036_reserved_pool_growth_gib,
            device_budget_gib=args.t036_device_budget_gib,
            exact_state_equality=not args.t036_break_state_equality,
            exact_token_equality=not args.t036_break_token_equality,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t036-a05-block-verifier-gate-{stamp}")
        write_t036_block_verifier_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "best_current_row": result["best_current_row"],
                    "best_conditional_projection_row": result["best_conditional_projection_row"],
                    "memory_contract": result["memory_contract"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t037_exact_suffix_prefill_gate:
        result = run_t037_exact_suffix_prefill_gate(
            block_sizes=args.t037_block_sizes,
            prefix_tokens=args.t037_prefix_tokens,
            vocab_size=args.t037_vocab_size,
            full64_suffix_prefill_seconds=args.t037_full64_suffix_prefill_seconds,
            extra_seconds=args.t037_extra_seconds,
            shadow_kv_gib=args.t037_shadow_kv_gib,
            scratch_gib=args.t037_scratch_gib,
            graph_memory_gib=args.t037_graph_memory_gib,
            dense_sidecar_gib=args.t037_dense_sidecar_gib,
            device_budget_gib=args.t037_device_budget_gib,
            timing_source=args.t037_timing_source,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t037-exact-suffix-prefill-gate-{stamp}")
        write_t037_exact_suffix_prefill_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "best_pricing_row": result["best_pricing_row"],
                    "memory_contract": result["memory_contract"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t037_direct_denominator_gate:
        result = run_t037_direct_denominator_gate(
            measured_direct_seconds_per_token=args.t037_direct_seconds_per_token,
            measured_kernel_seconds_per_token=args.t037_direct_kernel_seconds_per_token,
            measured_non_kernel_seconds_per_token=args.t037_direct_non_kernel_seconds_per_token,
            peak_cuda_gib=args.t037_direct_peak_cuda_gib,
            hidden_sidecar_gib=args.t037_direct_hidden_sidecar_gib,
            device_budget_gib=args.t037_device_budget_gib,
            exact_state_and_token=not args.t037_direct_break_exactness,
            setup_seconds_per_token=args.t037_direct_setup_seconds_per_token,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t037-direct-denominator-gate-{stamp}")
        write_t037_direct_denominator_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "best_current_row": result["best_current_row"],
                    "control_only_row": result["control_only_row"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t038_kernel_body_roofline_gate:
        result = run_t038_kernel_body_roofline_gate(
            resident_payload_gib=args.t038_resident_payload_gib,
            sustained_bandwidth_gib_s=args.t038_sustained_bandwidth_gib_s,
            hot_read_multiplier=args.t038_hot_read_multiplier,
            arithmetic_seconds_per_token=args.t038_arithmetic_seconds_per_token,
            control_seconds_per_token=args.t038_control_seconds_per_token,
            setup_seconds_per_token=args.t038_setup_seconds_per_token,
            peak_cuda_gib=args.t038_peak_cuda_gib,
            hidden_sidecar_gib=args.t038_hidden_sidecar_gib,
            device_budget_gib=args.t037_device_budget_gib,
            exact_storage_semantics=not args.t038_break_exact_storage_semantics,
            compact_affine_map_fraction=args.t038_compact_affine_map_fraction,
            sign_plane_map_fraction=args.t038_sign_plane_map_fraction,
            dense_map_fraction=args.t038_dense_map_fraction,
            decoded_dense_sidecar_gib=args.t038_decoded_dense_sidecar_gib,
            replacement_layout_gib=args.t038_replacement_layout_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t038-kernel-body-roofline-gate-{stamp}")
        write_t038_kernel_body_roofline_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "roofline": result["roofline"],
                    "candidate": result["scenarios"][-1],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t038_representative_map_gate:
        result = run_t038_representative_map_gate(
            representative_speedup_vs_current=args.t038_representative_speedup,
            exact_output_equivalence=not args.t038_representative_break_exactness,
            covers_full_body_fraction=args.t038_representative_coverage,
            hidden_sidecar_gib=args.t038_representative_hidden_sidecar_gib,
            peak_cuda_gib=args.t038_representative_peak_cuda_gib,
            device_budget_gib=args.t037_device_budget_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t038-representative-map-gate-{stamp}")
        write_t038_representative_map_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "representative_map": result["representative_map"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t038b_coverage_denominator_gate:
        result = run_t038b_coverage_denominator_gate(
            component_names=args.t038b_component_names,
            component_fractions=args.t038b_component_fractions,
            component_speedups=args.t038b_component_speedups,
            minimum_accelerated_body_fraction=args.t038b_minimum_accelerated_body_fraction,
            hidden_sidecar_gib=args.t038b_hidden_sidecar_gib,
            decoded_dense_sidecar_gib=args.t038b_decoded_dense_sidecar_gib,
            residual_fraction=args.t038b_residual_fraction,
            exact_arithmetic=not args.t038b_break_exact_arithmetic,
            exact_layout_semantics=not args.t038b_break_exact_layout_semantics,
            layout_replaces_storage=not args.t038b_break_layout_replaces_storage,
            peak_cuda_gib=args.t038b_peak_cuda_gib,
            device_budget_gib=args.t037_device_budget_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t038b-coverage-denominator-gate-{stamp}")
        write_t038b_coverage_denominator_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "coverage_model": result["coverage_model"],
                    "components": result["components"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t039_a01_exact_map_object_ledger_gate:
        result = run_t039_a01_exact_map_object_ledger_gate(
            component_names=args.t039_a01_component_names,
            component_fractions=args.t039_a01_component_fractions,
            component_speedups=args.t039_a01_component_speedups,
            object_equivalence_proven=args.t039_a01_object_equivalence_proven,
            representative_real_maps_audited=args.t039_a01_representative_real_maps_audited,
            residual_fraction=args.t039_a01_residual_fraction,
            hidden_sidecar_gib=args.t039_a01_hidden_sidecar_gib,
            decoded_dense_sidecar_gib=args.t039_a01_decoded_dense_sidecar_gib,
            offload_or_setup_gain_counted=args.t039_a01_offload_or_setup_gain_counted,
            layout_replaces_storage=args.t039_a01_layout_replaces_storage,
            exact_arithmetic_contract=not args.t039_a01_break_exact_arithmetic_contract,
            peak_cuda_gib=args.t039_a01_peak_cuda_gib,
            device_budget_gib=args.t037_device_budget_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t039-a01-exact-map-object-ledger-gate-{stamp}")
        write_t039_a01_exact_map_object_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "coverage_model": result["coverage_model"],
                    "object_audit": result["object_audit"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t039_a02_semantic_layout_ledger_gate:
        result = run_t039_a02_semantic_layout_ledger_gate(
            component_names=args.t039_a02_component_names,
            component_fractions=args.t039_a02_component_fractions,
            component_speedups=args.t039_a02_component_speedups,
            gauge_contracts_proven=args.t039_a02_gauge_contracts_proven,
            semantic_boundaries_priced=args.t039_a02_semantic_boundaries_priced,
            forced_canonical_fraction=args.t039_a02_forced_canonical_fraction,
            residual_fraction=args.t039_a02_residual_fraction,
            hidden_sidecar_gib=args.t039_a02_hidden_sidecar_gib,
            decoded_dense_sidecar_gib=args.t039_a02_decoded_dense_sidecar_gib,
            offload_or_setup_gain_counted=args.t039_a02_offload_or_setup_gain_counted,
            layout_replaces_storage=args.t039_a02_layout_replaces_storage,
            exact_arithmetic_contract=not args.t039_a02_break_exact_arithmetic_contract,
            peak_cuda_gib=args.t039_a02_peak_cuda_gib,
            device_budget_gib=args.t037_device_budget_gib,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t039-a02-semantic-layout-ledger-gate-{stamp}")
        write_t039_a02_semantic_layout_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "coverage_model": result["coverage_model"],
                    "layout_audit": result["layout_audit"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t039_real_map_object_audit:
        result = run_t039_real_map_object_audit(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            tensor_names=args.t039_real_map_audit_tensors,
            block_rows=args.t039_real_map_audit_rows,
            block_cols=args.t039_real_map_audit_cols,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t039-real-map-object-audit-{stamp}")
        write_t039_real_map_object_audit_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "audited_blocks": result["audited_blocks"],
                    "ledger_coverage": result["ledger"]["coverage_model"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t039b_a01_non_row_exact_object_ledger:
        result = run_t039b_a01_non_row_exact_object_extraction_ledger(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            tensor_names=args.t039_real_map_audit_tensors,
            block_rows=args.t039_real_map_audit_rows,
            block_cols=args.t039_real_map_audit_cols,
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t039b-a01-non-row-exact-object-ledger-{stamp}")
        write_t039b_a01_non_row_exact_object_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "best_object_family_row": result["best_object_family_row"],
                    "ledger_coverage": result["ledger"]["coverage_model"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t039b_a02_gauge_sat_ledger:
        result = run_t039b_a02_gauge_sat_canonical_tail_ledger()
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t039b-a02-gauge-sat-ledger-{stamp}")
        write_t039b_a02_gauge_sat_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "gauge_sat": result["gauge_sat"],
                    "ledger_coverage": result["ledger"]["coverage_model"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t005_real_weight_sentinel:
        result = run_t005_real_weight_sentinel(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            sample_rows=args.sentinel_sample_rows,
            sample_cols=args.sentinel_sample_cols,
            min_required_tensors=args.sentinel_min_required_tensors,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t005-real-weight-sentinel-{stamp}")
        write_t005_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t006_real_residency_block_smoke:
        result = run_t006_real_residency_block_smoke(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            block_rows=args.block_smoke_rows,
            block_cols=args.block_smoke_cols,
            batch=args.block_smoke_batch,
            min_required_blocks=args.block_smoke_min_required,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t006-real-residency-block-smoke-{stamp}")
        write_t006_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t007_real_layer_smoke:
        result = run_t007_real_layer_residency_smoke(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            layer_index=args.layer_smoke_index,
            hidden_cols=args.layer_smoke_hidden_cols,
            attention_width=args.layer_smoke_attention_width,
            mlp_width=args.layer_smoke_mlp_width,
            sequence_tokens=args.layer_smoke_sequence_tokens,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t007-real-layer-smoke-{stamp}")
        write_t007_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "layer_receipt": result["layer_receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t008_compressed_generation_smoke:
        result = run_t008_compressed_generation_smoke(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            layer_indices=args.generation_layer_indices,
            hidden_cols=args.generation_hidden_cols,
            attention_width=args.generation_attention_width,
            mlp_width=args.generation_mlp_width,
            vocab_rows=args.generation_vocab_rows,
            prompt_tokens=args.generation_prompt_tokens,
            max_new_tokens=args.generation_max_new_tokens,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t008-compressed-generation-smoke-{stamp}")
        write_t008_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "generation_receipt": result["generation_receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t009_t013_runtime_scaler:
        result = run_t009_t013_runtime_scaler(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            full_layer_index=args.runtime_full_layer_index,
            scale_layer_indices=args.runtime_scale_layer_indices,
            scale_row_limit=args.runtime_scale_row_limit,
            projection_batch=args.runtime_projection_batch,
            prompt_text=args.runtime_prompt_text,
            require_tokenizer=not args.runtime_allow_tokenizer_fallback,
            kv_context_tokens=args.runtime_kv_context_tokens,
            kv_effective_bits=args.runtime_kv_effective_bits,
            exact_recent_tokens=args.runtime_exact_recent_tokens,
            bandwidth_gib_per_second=args.runtime_bandwidth_gib_per_second,
            runtime_overhead_gib=args.runtime_scaler_overhead_gib,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t009-t013-runtime-scaler-{stamp}")
        write_t009_t013_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "t009": {key: value for key, value in result["t009_full_width_layer"].items() if key != "projection_receipts"},
                    "t010": {key: value for key, value in result["t010_multilayer_scale"].items() if key != "projection_receipts"},
                    "t011": result["t011_tokenizer_prompt"],
                    "t012": result["t012_kv_cache_ledger"],
                    "t013": result["t013_full_model_equivalent_pricing"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t014_denominator_tightening:
        result = run_t014_denominator_tightening(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            mlp_layer_index=args.t014_mlp_layer_index,
            mlp_row_limit=None if args.t014_mlp_row_limit <= 0 else args.t014_mlp_row_limit,
            projection_batch=args.t014_projection_batch,
            target_mlp_bits=args.t014_target_mlp_bits,
            context_tokens=args.t014_context_tokens,
            bandwidth_gib_per_second=args.t014_bandwidth_gib_per_second,
            include_stretch=not args.t014_no_stretch,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t014-denominator-tightening-{stamp}")
        write_t014_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "mlp_projection_gate": {key: value for key, value in result["mlp_projection_gate"].items() if key != "projection_receipts"},
                    "packed_arena_receipt": result["packed_arena_receipt"],
                    "candidate_table": [
                        {
                            key: candidate[key]
                            for key in (
                                "name",
                                "total_with_kv_gib",
                                "slack_vs_12gib",
                                "saves_vs_current_t013_gib",
                                "tokens_per_second_proxy",
                                "speedup_vs_q4_fp16kv",
                                "candidate_fits_12gib",
                                "status",
                            )
                        }
                        for candidate in result["candidate_table"]
                    ],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t015_packed_qwen_inference:
        result = run_t015_packed_qwen_inference(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t015_prompt_text,
            max_new_tokens=args.t015_max_new_tokens,
            layer_limit=None if args.t015_layer_limit <= 0 else args.t015_layer_limit,
            device=args.t015_device,
            target_mlp_bits=args.t015_target_mlp_bits,
            lm_head_chunk_rows=args.t015_lm_head_chunk_rows,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t015-packed-qwen-inference-{stamp}")
        write_t015_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t016_persistent_packed_cache:
        result = run_t016_persistent_packed_cache(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t016_prompt_text,
            max_new_tokens=args.t016_max_new_tokens,
            layer_limit=None if args.t016_layer_limit <= 0 else args.t016_layer_limit,
            device=args.t016_device,
            target_mlp_bits=args.t016_target_mlp_bits,
            lm_head_chunk_rows=args.t016_lm_head_chunk_rows,
            cache_dir=args.t016_cache_dir,
            refresh_cache=args.t016_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t016-persistent-packed-cache-{stamp}")
        write_t016_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t017_packed_linear_kernel:
        result = run_t017_packed_linear_kernel_gate(
            rows=args.t017_rows,
            cols=args.t017_cols,
            tokens=args.t017_tokens,
            repair_fraction=args.t017_repair_fraction,
            iterations=args.t017_iterations,
            seed=args.t017_seed,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t017-packed-linear-kernel-{stamp}")
        write_t017_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t017_kernel_qwen_inference:
        result = run_t017_kernel_qwen_inference(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
            device_credit_flow=args.t017_device_credit_flow,
            packed_work_coalescing=args.t017_packed_work_coalescing,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t017-kernel-qwen-inference-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t018_resident_payload_qwen_inference:
        result = run_t018_resident_payload_qwen_inference(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
            preload_payloads=not args.t018_no_preload_payloads,
            device_credit_flow=args.t017_device_credit_flow,
            packed_work_coalescing=args.t017_packed_work_coalescing,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t018-resident-payload-qwen-inference-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t019_binary_payload_qwen_inference:
        result = run_t019_binary_payload_qwen_inference(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
            device_credit_flow=args.t017_device_credit_flow,
            packed_work_coalescing=args.t017_packed_work_coalescing,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t019-binary-payload-qwen-inference-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_cumulative_execution_shape:
        result = run_t020_cumulative_execution_shape_gate(
            min_individual_speedup=args.t020_min_speedup,
            min_cumulative_speedup=2.0,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-cumulative-execution-shape-{stamp}")
        write_t020_cumulative_execution_shape_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "composition": result["composition"],
                    "numeric_items": result["numeric_items"],
                    "structural_items": result["structural_items"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t021_runtime_obligation_measure:
        result = run_t021_runtime_obligation_measure_gate()
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t021-runtime-obligation-measure-{stamp}")
        write_t021_report(result, out_dir, "T021 Runtime Obligation Measure Gate")
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t021_transaction_receipt_gate:
        result = run_t021_transaction_receipt_gate()
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t021-transaction-receipt-gate-{stamp}")
        write_t021_report(result, out_dir, "T021 Transaction Receipt Gate")
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t021_page_activation_certificate:
        result = run_t021_page_activation_certificate_gate()
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t021-page-activation-certificate-{stamp}")
        write_t021_report(result, out_dir, "T021 Page Activation Certificate Gate")
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t021_cumulative_decoder_runtime_path:
        result = run_t021_cumulative_decoder_runtime_path_gate()
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t021-cumulative-decoder-runtime-path-{stamp}")
        write_t021_report(result, out_dir, "T021 Cumulative Decoder Runtime Path Gate")
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t021_qwen_runtime:
        result = run_t021_qwen_runtime(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t023_device_payload_budget_gib,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t021-qwen-runtime-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "t021_cumulative_gate": result["t021_cumulative_gate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t022_qwen_runtime:
        result = run_t022_qwen_runtime(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t022_device_payload_budget_gib,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t022-qwen-runtime-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "t022_stack": result["t022_stack"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t023_qwen_runtime:
        result = run_t023_qwen_runtime(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t023-qwen-runtime-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "t023_stack": result["t023_stack"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t024_qwen_runtime:
        result = run_t024_qwen_runtime(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t024-qwen-runtime-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": result["receipt"],
                    "t024_stack": result["t024_stack"],
                    "t024_exact_quotients": result["t024_exact_quotients"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t025_qwen_runtime:
        result = run_t025_qwen_runtime(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t025_device_payload_budget_gib,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t025-qwen-runtime-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": _runtime_receipt_console_summary(result["receipt"]),
                    "t025_stack": result["t025_stack"],
                    "t025_exact_quotients": result["t025_exact_quotients"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t026_qwen_runtime:
        result = run_t026_qwen_runtime(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t026_device_payload_budget_gib,
            packed_linear_block_m=args.t026_packed_linear_block_m,
            packed_linear_block_c=args.t026_packed_linear_block_c,
            packed_linear_block_r=args.t026_packed_linear_block_r,
            fused_mlp_gate_up_kernel=args.t026_fused_mlp_gate_up_kernel,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t026-qwen-runtime-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": _runtime_receipt_console_summary(result["receipt"]),
                    "t026_stack": result["t026_stack"],
                    "t026_exact_quotients": result["t026_exact_quotients"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t027_qwen_runtime:
        result = run_t027_qwen_runtime(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t027_device_payload_budget_gib,
            packed_linear_block_m=args.t026_packed_linear_block_m,
            packed_linear_block_c=args.t026_packed_linear_block_c,
            packed_linear_block_r=args.t026_packed_linear_block_r,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t027-qwen-runtime-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": _runtime_receipt_console_summary(result["receipt"]),
                    "t027_stack": result["t027_stack"],
                    "t027_exact_quotients": result["t027_exact_quotients"],
                    "t027_projection": result["t027_projection"],
                    "t027_speed_comparison": result["t027_speed_comparison"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t028_native_generation_speed:
        result = run_t028_native_generation_speed(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t028_device_payload_budget_gib,
            packed_linear_block_m=args.t028_packed_linear_block_m,
            packed_linear_block_c=args.t028_packed_linear_block_c,
            packed_linear_block_r=args.t028_packed_linear_block_r,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t028-native-generation-speed-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": _runtime_receipt_console_summary(result["receipt"]),
                    "t028_stack": result["t028_stack"],
                    "t028_gates": result["t028_gates"],
                    "t028_projection": result["t028_projection"],
                    "t028_speed_comparison": result["t028_speed_comparison"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t029_speed_gpu_efficiency:
        result = run_t029_speed_gpu_efficiency(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t029_device_payload_budget_gib,
            packed_linear_block_m=args.t029_packed_linear_block_m,
            packed_linear_block_c=args.t029_packed_linear_block_c,
            packed_linear_block_r=args.t029_packed_linear_block_r,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t029-speed-gpu-efficiency-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": _runtime_receipt_console_summary(result["receipt"]),
                    "t029_stack": result["t029_stack"],
                    "t029_memory_audit": result["t029_memory_audit"],
                    "t029_speed_efficiency_comparison": result["t029_speed_efficiency_comparison"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t030_speed_gpu_efficiency:
        result = run_t030_speed_gpu_efficiency(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t030_device_payload_budget_gib,
            packed_linear_block_m=args.t030_packed_linear_block_m,
            packed_linear_block_c=args.t030_packed_linear_block_c,
            packed_linear_block_r=args.t030_packed_linear_block_r,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
            fused_qkv_compact_projection_kernel=args.t030_enable_fused_qkv_kernel,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t030-speed-gpu-efficiency-{stamp}")
        write_t017_qwen_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "receipt": _runtime_receipt_console_summary(result["receipt"]),
                    "t030_stack": result["t030_stack"],
                    "t030_gate_audit": result["t030_gate_audit"],
                    "t030_memory_audit": result["t030_memory_audit"],
                    "t030_speed_efficiency_comparison": result["t030_speed_efficiency_comparison"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t031_macro_architecture_stack:
        result = run_t031_macro_architecture_stack(
            model_id=args.qwen_model_id,
            local_model_dir=args.local_model_dir,
            allow_hf_config_fetch=not args.no_hf_config_fetch,
            prompt_text=args.t017_prompt_text,
            max_new_tokens=args.t017_max_new_tokens,
            layer_limit=None if args.t017_layer_limit <= 0 else args.t017_layer_limit,
            device=args.t017_device,
            target_mlp_bits=args.t017_target_mlp_bits,
            lm_head_chunk_rows=args.t017_lm_head_chunk_rows,
            device_payload_budget_gib=args.t030_device_payload_budget_gib,
            packed_linear_block_m=args.t030_packed_linear_block_m,
            packed_linear_block_c=args.t030_packed_linear_block_c,
            packed_linear_block_r=args.t030_packed_linear_block_r,
            cache_dir=args.t017_cache_dir,
            refresh_cache=args.t017_refresh_cache,
            fused_qkv_compact_projection_kernel=args.t030_enable_fused_qkv_kernel,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t031-macro-architecture-stack-{stamp}")
        write_t031_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "speed_projection": result["candidate"]["speed_projection"],
                    "memory_projection": result["candidate"]["memory_projection"],
                    "runtime_comparison": result["candidate"]["runtime_comparison"],
                    "mechanism_count": result["candidate"]["mechanism_count"],
                    "real_runtime_missing_ids": result["readiness"]["t031_real_runtime_missing_ids"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_device_credit_flow:
        result = run_t020_device_credit_flow_gate(
            baseline_total_seconds=args.t020_baseline_total_seconds,
            baseline_payload_load_seconds=args.t020_baseline_payload_load_seconds,
            baseline_kernel_seconds=args.t020_baseline_kernel_seconds,
            baseline_tokens_per_second=args.t020_baseline_tokens_per_second,
            device_budget_gib=args.t020_device_budget_gib,
            baseline_denominator_gib=args.t020_baseline_denominator_gib,
            staged_payload_window_mib=args.t020_staged_payload_window_mib,
            accumulator_window_mib=args.t020_accumulator_window_mib,
            credit_state_mib=args.t020_credit_state_mib,
            sync_workspace_mib=args.t020_sync_workspace_mib,
            decode_seconds=args.t020_decode_seconds,
            epilogue_seconds=args.t020_epilogue_seconds,
            credit_overhead_fraction=args.t020_credit_overhead_fraction,
            sync_penalty_seconds=args.t020_sync_penalty_seconds,
            min_speedup=args.t020_min_speedup,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-device-credit-flow-{stamp}")
        write_t020_device_credit_flow_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_packed_work_coalescing:
        result = run_t020_packed_work_coalescing_gate(
            layers=args.t020_coalescing_layers,
            hidden_size=args.t020_coalescing_hidden_size,
            tokens=args.t020_coalescing_tokens,
            activation_bytes=args.t020_coalescing_activation_bytes,
            min_legal_outputs_per_block=args.t020_coalescing_min_legal_outputs,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-packed-work-coalescing-{stamp}")
        write_t020_packed_work_coalescing_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "legal_boundaries": result["legal_boundaries"],
                    "traffic": result["traffic"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_execution_order_slab:
        result = run_t020_execution_order_slab_gate(
            baseline_objects=args.t020_slab_baseline_objects,
            baseline_layer1_acquisition_seconds=args.t020_slab_baseline_acquisition_seconds,
            fixed_object_fraction=args.t020_slab_fixed_object_fraction,
            offset_walk_nanoseconds=args.t020_slab_offset_walk_nanoseconds,
            min_acquisition_speedup=args.t020_slab_min_acquisition_speedup,
            lm_head_rows=args.t020_slab_lm_head_rows,
            lm_head_chunk_rows=args.t020_slab_lm_head_chunk_rows,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-execution-order-slab-{stamp}")
        write_t020_execution_order_slab_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_immutable_arena_receipt:
        result = run_t020_immutable_arena_receipt_gate(
            baseline_dynamic_discovery_seconds=args.t020_receipt_baseline_dynamic_discovery_seconds,
            expected_tokens_between_rebuilds=args.t020_receipt_expected_tokens_between_rebuilds,
            offline_build_seconds=args.t020_receipt_offline_build_seconds,
            open_arena_seconds=args.t020_receipt_open_arena_seconds,
            descriptor_verify_microseconds=args.t020_receipt_descriptor_verify_microseconds,
            root_lookup_seconds=args.t020_receipt_root_lookup_seconds,
            min_speedup=args.t020_receipt_min_speedup,
            lm_head_rows=args.t020_receipt_lm_head_rows,
            lm_head_chunk_rows=args.t020_receipt_lm_head_chunk_rows,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-immutable-arena-receipt-{stamp}")
        write_t020_immutable_arena_receipt_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "verification": result["verification"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_conversion_time_arena_compiler:
        result = run_t020_conversion_time_arena_compiler_gate(
            baseline_visible_denominator_seconds=args.t020_compiler_baseline_visible_denominator_seconds,
            receipt_admission_seconds=args.t020_compiler_receipt_admission_seconds,
            subset_descriptors=args.t020_compiler_subset_descriptors,
            min_saved_fraction=args.t020_compiler_min_saved_fraction,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-conversion-time-arena-compiler-{stamp}")
        write_t020_conversion_time_arena_compiler_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "descriptor_equality": result["descriptor_equality"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_affine_packed_view:
        result = run_t020_affine_packed_view_gate(
            rows=args.t020_affine_rows,
            cols=args.t020_affine_cols,
            group_cols=args.t020_affine_group_cols,
            visible_denominator_seconds=args.t020_affine_visible_denominator_seconds,
            materialization_cost_removed_seconds=args.t020_affine_materialization_cost_removed_seconds,
            affine_decode_overhead_seconds=args.t020_affine_decode_overhead_seconds,
            coalescing_penalty_seconds=args.t020_affine_coalescing_penalty_seconds,
            min_saved_fraction=args.t020_affine_min_saved_fraction,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-affine-packed-view-{stamp}")
        write_t020_affine_packed_view_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "domain": result["domain"],
                    "equality": result["equality"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_recompute_on_reject:
        result = run_t020_recompute_on_reject_gate(
            branch_depth=args.t020_recompute_branch_depth,
            branch_width=args.t020_recompute_branch_width,
            base_acceptance=args.t020_recompute_base_acceptance,
            kv_write_cost=args.t020_recompute_kv_write_cost,
            memory_pressure_cost=args.t020_recompute_memory_pressure_cost,
            depth_memory_pressure_step=args.t020_recompute_depth_memory_pressure_step,
            recompute_cost=args.t020_recompute_cost,
            cleanup_cost=args.t020_recompute_cleanup_cost,
            receipt_cost=args.t020_recompute_receipt_cost,
            max_store_fraction=args.t020_recompute_max_store_fraction,
            min_speedup=args.t020_recompute_min_speedup,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-recompute-on-reject-{stamp}")
        write_t020_recompute_on_reject_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_expected_accepted_span:
        result = run_t020_expected_accepted_span_gate(
            risk_margin=args.t020_span_risk_margin,
            min_speedup=args.t020_span_min_speedup,
            base_span_cost=args.t020_span_base_cost,
            verify_cost_per_token=args.t020_span_verify_cost_per_token,
            delayed_kv_cost_per_token=args.t020_span_delayed_kv_cost_per_token,
            lm_head_cost_per_token=args.t020_span_lm_head_cost_per_token,
            rollback_cost_per_rejected_token=args.t020_span_rollback_cost_per_rejected_token,
            schedule_gap_cost=args.t020_span_schedule_gap_cost,
            graph_specialization_cost=args.t020_span_graph_specialization_cost,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-expected-accepted-span-{stamp}")
        write_t020_expected_accepted_span_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_prefix_transaction_ledger:
        result = run_t020_prefix_transaction_ledger_gate(
            layers=args.t020_prefix_layers,
            heads=args.t020_prefix_heads,
            branch_length=args.t020_prefix_branch_length,
            committed_tokens=args.t020_prefix_committed_tokens,
            min_speedup=args.t020_prefix_min_speedup,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-prefix-transaction-ledger-{stamp}")
        write_t020_prefix_transaction_ledger_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "layout": {
                        "layers": result["layout"]["layers"],
                        "heads": result["layout"]["heads"],
                        "branch_length": result["layout"]["branch_length"],
                        "committed_tokens": result["layout"]["committed_tokens"],
                        "delta_entries": result["layout"]["delta_entries"],
                        "case_count": len(result["layout"]["cases"]),
                    },
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_acceptance_weighted_kv_materialization:
        result = run_t020_acceptance_weighted_kv_materialization_gate(
            accepted_span_probability=args.t020_materialization_accepted_span_probability,
            recompute_to_kv_ratio=args.t020_materialization_recompute_to_kv_ratio,
            receipt_cost_per_token_layer=args.t020_materialization_receipt_cost_per_token_layer,
            conditional_schedule_cost=args.t020_materialization_conditional_schedule_cost,
            min_speedup=args.t020_materialization_min_speedup,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-acceptance-weighted-kv-materialization-{stamp}")
        write_t020_acceptance_weighted_kv_materialization_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "domain": result["domain"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "exactness": result["exactness"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_basis_shared_kv_transform:
        result = run_t020_basis_shared_kv_transform_gate(
            vector_size=args.t020_basis_vector_size,
            epsilon_numeric=args.t020_basis_epsilon_numeric,
            attention_margin=args.t020_basis_attention_margin,
            min_speedup=args.t020_basis_min_speedup,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-basis-shared-kv-transform-{stamp}")
        write_t020_basis_shared_kv_transform_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "transform_class": result["transform_class"],
                    "proof": result["proof"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return
    if args.t020_ring_exact_tail_receipts:
        result = run_t020_ring_exact_tail_receipts_gate(
            full_context_tokens=args.t020_ring_full_context_tokens,
            min_exact_ring_tokens=args.t020_ring_min_exact_tokens,
            min_speedup=args.t020_ring_min_speedup,
        )
        result["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = args.out_dir or (DEFAULT_OUT / f"t020-ring-exact-tail-receipts-{stamp}")
        write_t020_ring_exact_tail_receipts_report(result, out_dir)
        print(result["activation"])
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "readiness": result["readiness"],
                    "budget": result["budget"],
                    "baseline": result["baseline"],
                    "candidate": result["candidate"],
                    "patterns": result["patterns"],
                    "blockers": result["blockers"],
                    "next_actions": result["next_actions"],
                },
                indent=2,
            )
        )
        return

    result = run_probe(args)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT / stamp)
    write_report(result, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
