"""T038 kernel/body denominator roofline gates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


ACTIVATION = "T038_KERNEL_BODY_ROOFLINE_GATE_ACTIVE"
REPRESENTATIVE_MAP_ACTIVATION = "T038_REPRESENTATIVE_MAP_GATE_ACTIVE"
T038B_COVERAGE_ACTIVATION = "T038B_COVERAGE_DENOMINATOR_GATE_ACTIVE"

TARGET_TOKENS_PER_SECOND = 15.0
TARGET_SECONDS_PER_TOKEN = 1.0 / TARGET_TOKENS_PER_SECOND
R452_KERNEL_SECONDS_PER_TOKEN = 0.5049169003032148
R452_NON_KERNEL_SECONDS_PER_TOKEN = 0.5726308997254819
R452_SECONDS_PER_TOKEN = 1.0775478000286967
R452_TOKENS_PER_SECOND = 0.9280330765589875
R452_DEVICE_RESIDENT_GIB = 10.458633184432983
R452_PEAK_CUDA_GIB = 10.652063369750977
DEFAULT_SUSTAINED_BANDWIDTH_GIB_S = 514.0
DEFAULT_DEVICE_BUDGET_GIB = 12.0
T017_MLP_LIKE_SYNTHETIC_SPEEDUP = 4.7723


def _validate_component_inputs(
    *,
    component_names: tuple[str, ...],
    component_fractions: tuple[float, ...],
    component_speedups: tuple[float, ...],
) -> list[str]:
    errors: list[str] = []
    if len(component_names) != len(component_fractions) or len(component_fractions) != len(component_speedups):
        errors.append("component_name_fraction_speedup_length_mismatch")
    if not component_fractions:
        errors.append("no_body_components_priced")
    if any(fraction < 0.0 for fraction in component_fractions):
        errors.append("negative_body_component_fraction")
    if any(speedup <= 0.0 for speedup in component_speedups):
        errors.append("nonpositive_body_component_speedup")
    return errors


def _roofline_row(
    *,
    name: str,
    resident_payload_gib: float,
    hot_read_multiplier: float,
    sustained_bandwidth_gib_s: float,
    arithmetic_seconds_per_token: float,
    control_seconds_per_token: float,
    setup_seconds_per_token: float,
    peak_cuda_gib: float,
    device_budget_gib: float,
    hidden_sidecar_gib: float,
    exact_storage_semantics: bool,
) -> dict[str, Any]:
    hot_bytes_gib = resident_payload_gib * hot_read_multiplier
    byte_seconds = hot_bytes_gib / sustained_bandwidth_gib_s if sustained_bandwidth_gib_s > 0.0 else float("inf")
    total_seconds = byte_seconds + arithmetic_seconds_per_token + control_seconds_per_token + setup_seconds_per_token
    tokens_per_second = 1.0 / total_seconds if total_seconds > 0.0 else 0.0
    target_pass = (
        total_seconds <= TARGET_SECONDS_PER_TOKEN
        and peak_cuda_gib <= device_budget_gib
        and hidden_sidecar_gib == 0.0
        and exact_storage_semantics
    )
    return {
        "name": name,
        "resident_payload_gib": resident_payload_gib,
        "hot_read_multiplier": hot_read_multiplier,
        "hot_bytes_gib": hot_bytes_gib,
        "sustained_bandwidth_gib_s": sustained_bandwidth_gib_s,
        "byte_seconds_per_token": byte_seconds,
        "arithmetic_seconds_per_token": arithmetic_seconds_per_token,
        "control_seconds_per_token": control_seconds_per_token,
        "setup_seconds_per_token": setup_seconds_per_token,
        "total_seconds_per_token": total_seconds,
        "tokens_per_second": tokens_per_second,
        "peak_cuda_gib": peak_cuda_gib,
        "device_budget_gib": device_budget_gib,
        "hidden_sidecar_gib": hidden_sidecar_gib,
        "exact_storage_semantics": exact_storage_semantics,
        "target_pass": target_pass,
        "target_seconds_per_token": TARGET_SECONDS_PER_TOKEN,
    }


def run_t038_kernel_body_roofline_gate(
    *,
    resident_payload_gib: float = R452_DEVICE_RESIDENT_GIB,
    sustained_bandwidth_gib_s: float = DEFAULT_SUSTAINED_BANDWIDTH_GIB_S,
    hot_read_multiplier: float = 1.0,
    arithmetic_seconds_per_token: float = 0.0,
    control_seconds_per_token: float = 0.0,
    setup_seconds_per_token: float = 0.0,
    peak_cuda_gib: float = R452_PEAK_CUDA_GIB,
    hidden_sidecar_gib: float = 0.0,
    device_budget_gib: float = DEFAULT_DEVICE_BUDGET_GIB,
    exact_storage_semantics: bool = True,
    compact_affine_map_fraction: float = 1.0,
    sign_plane_map_fraction: float = 0.0,
    dense_map_fraction: float = 0.0,
    decoded_dense_sidecar_gib: float = 0.0,
    replacement_layout_gib: float | None = None,
) -> dict[str, Any]:
    """Price the T038 body-denominator preconditions.

    The gate decides whether the next kernel/body research path is physically
    admissible. It does not claim a measured production speedup.
    """

    current_effective_bandwidth = R452_DEVICE_RESIDENT_GIB / R452_KERNEL_SECONDS_PER_TOKEN
    required_one_read_bandwidth = resident_payload_gib / TARGET_SECONDS_PER_TOKEN
    max_reads_at_target = sustained_bandwidth_gib_s * TARGET_SECONDS_PER_TOKEN / max(resident_payload_gib, 1e-12)
    candidate = _roofline_row(
        name="candidate_weight_once_body",
        resident_payload_gib=resident_payload_gib,
        hot_read_multiplier=hot_read_multiplier,
        sustained_bandwidth_gib_s=sustained_bandwidth_gib_s,
        arithmetic_seconds_per_token=arithmetic_seconds_per_token,
        control_seconds_per_token=control_seconds_per_token,
        setup_seconds_per_token=setup_seconds_per_token,
        peak_cuda_gib=peak_cuda_gib,
        device_budget_gib=device_budget_gib,
        hidden_sidecar_gib=hidden_sidecar_gib + decoded_dense_sidecar_gib,
        exact_storage_semantics=exact_storage_semantics,
    )
    current = _roofline_row(
        name="r452_current_body",
        resident_payload_gib=R452_DEVICE_RESIDENT_GIB,
        hot_read_multiplier=1.0,
        sustained_bandwidth_gib_s=current_effective_bandwidth,
        arithmetic_seconds_per_token=0.0,
        control_seconds_per_token=R452_NON_KERNEL_SECONDS_PER_TOKEN,
        setup_seconds_per_token=0.0,
        peak_cuda_gib=R452_PEAK_CUDA_GIB,
        device_budget_gib=device_budget_gib,
        hidden_sidecar_gib=0.0,
        exact_storage_semantics=True,
    )
    control_removed = _roofline_row(
        name="r452_control_removed_body",
        resident_payload_gib=R452_DEVICE_RESIDENT_GIB,
        hot_read_multiplier=1.0,
        sustained_bandwidth_gib_s=current_effective_bandwidth,
        arithmetic_seconds_per_token=0.0,
        control_seconds_per_token=0.0,
        setup_seconds_per_token=0.0,
        peak_cuda_gib=R452_PEAK_CUDA_GIB,
        device_budget_gib=device_budget_gib,
        hidden_sidecar_gib=0.0,
        exact_storage_semantics=True,
    )
    layout_gib = resident_payload_gib if replacement_layout_gib is None else float(replacement_layout_gib)
    layout_replaces_storage = layout_gib <= resident_payload_gib and decoded_dense_sidecar_gib == 0.0
    map_fraction_total = compact_affine_map_fraction + sign_plane_map_fraction + dense_map_fraction
    format_covers_body = (
        abs(map_fraction_total - 1.0) <= 1e-6
        and compact_affine_map_fraction + sign_plane_map_fraction >= 0.80
        and dense_map_fraction <= 0.20
        and exact_storage_semantics
    )
    roofline_survives = candidate["target_pass"]
    gate_ready = roofline_survives and layout_replaces_storage and format_covers_body

    blockers: list[str] = []
    if not roofline_survives:
        blockers.append("hot_byte_roofline_does_not_reach_15_tok_s")
    if candidate["hot_read_multiplier"] > max_reads_at_target:
        blockers.append("hot_read_multiplier_exceeds_target_budget")
    if not exact_storage_semantics:
        blockers.append("exact_storage_semantics_not_proven")
    if hidden_sidecar_gib + decoded_dense_sidecar_gib > 0.0:
        blockers.append("hidden_or_decoded_dense_sidecar_live")
    if peak_cuda_gib > device_budget_gib:
        blockers.append("projected_peak_cuda_exceeds_device_budget")
    if not layout_replaces_storage:
        blockers.append("replacement_layout_does_not_replace_resident_storage")
    if not format_covers_body:
        blockers.append("resident_format_does_not_cover_enough_body_maps")

    return {
        "probe": "t038_kernel_body_roofline_gate",
        "activation": ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": {
            "tokens_per_second": TARGET_TOKENS_PER_SECOND,
            "seconds_per_token": TARGET_SECONDS_PER_TOKEN,
        },
        "baseline": {
            "source": "R452/T028 plus T037 direct denominator",
            "tokens_per_second": R452_TOKENS_PER_SECOND,
            "seconds_per_token": R452_SECONDS_PER_TOKEN,
            "kernel_seconds_per_token": R452_KERNEL_SECONDS_PER_TOKEN,
            "non_kernel_seconds_per_token": R452_NON_KERNEL_SECONDS_PER_TOKEN,
            "device_resident_gib": R452_DEVICE_RESIDENT_GIB,
            "peak_cuda_gib": R452_PEAK_CUDA_GIB,
            "current_effective_bandwidth_gib_s": current_effective_bandwidth,
        },
        "roofline": {
            "required_one_read_bandwidth_gib_s": required_one_read_bandwidth,
            "sustained_bandwidth_gib_s": sustained_bandwidth_gib_s,
            "max_hot_reads_at_target": max_reads_at_target,
            "required_kernel_reduction_if_control_zero": R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN,
        },
        "scenarios": [current, control_removed, candidate],
        "format_audit": {
            "compact_affine_map_fraction": compact_affine_map_fraction,
            "sign_plane_map_fraction": sign_plane_map_fraction,
            "dense_map_fraction": dense_map_fraction,
            "map_fraction_total": map_fraction_total,
            "format_covers_body": format_covers_body,
            "exact_storage_semantics": exact_storage_semantics,
        },
        "layout_audit": {
            "resident_payload_gib": resident_payload_gib,
            "replacement_layout_gib": layout_gib,
            "decoded_dense_sidecar_gib": decoded_dense_sidecar_gib,
            "hidden_sidecar_gib": hidden_sidecar_gib,
            "layout_replaces_storage": layout_replaces_storage,
        },
        "readiness": {
            "t038_kernel_body_roofline_gate_ready": gate_ready,
            "t038_hot_byte_roofline_survives": roofline_survives,
            "t038_resident_format_covers_body": format_covers_body,
            "t038_layout_replaces_storage": layout_replaces_storage,
            "t038_exact_storage_semantics": exact_storage_semantics,
            "t038_no_hidden_sidecar": hidden_sidecar_gib + decoded_dense_sidecar_gib == 0.0,
            "t038_fits_device_budget": peak_cuda_gib <= device_budget_gib,
            "t038_production_implementation_allowed": False,
        },
        "blockers": blockers,
        "next_actions": [
            "Implement a representative exact large-map body gate before whole-token VM work.",
            "Instrument real hot-read multiplier and arithmetic time instead of keeping default roofline assumptions.",
        ]
        if gate_ready
        else [
            "Do not implement the whole-token VM until the roofline, format, and layout blockers are removed.",
            "If blockers are structural rather than instrumentation gaps, open a new research round with a different body representation.",
        ],
    }


def run_t038_representative_map_gate(
    *,
    representative_speedup_vs_current: float = T017_MLP_LIKE_SYNTHETIC_SPEEDUP,
    exact_output_equivalence: bool = True,
    covers_full_body_fraction: float = 0.80,
    hidden_sidecar_gib: float = 0.0,
    peak_cuda_gib: float = R452_PEAK_CUDA_GIB,
    device_budget_gib: float = DEFAULT_DEVICE_BUDGET_GIB,
) -> dict[str, Any]:
    """Check whether a representative exact map can carry the T038 denominator."""

    required_speedup = R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN
    projected_kernel_seconds = R452_KERNEL_SECONDS_PER_TOKEN / max(representative_speedup_vs_current, 1e-12)
    projected_direct_seconds_if_control_zero = projected_kernel_seconds
    projected_tokens_per_second = (
        1.0 / projected_direct_seconds_if_control_zero if projected_direct_seconds_if_control_zero > 0.0 else 0.0
    )
    speed_pass = representative_speedup_vs_current >= required_speedup
    coverage_pass = covers_full_body_fraction >= 0.80
    gate_ready = (
        speed_pass
        and exact_output_equivalence
        and coverage_pass
        and hidden_sidecar_gib == 0.0
        and peak_cuda_gib <= device_budget_gib
    )

    blockers: list[str] = []
    if not speed_pass:
        blockers.append("representative_map_speedup_below_required_7_5738x")
    if not exact_output_equivalence:
        blockers.append("representative_map_exact_output_not_proven")
    if not coverage_pass:
        blockers.append("representative_map_does_not_cover_enough_body")
    if hidden_sidecar_gib > 0.0:
        blockers.append("representative_map_hidden_sidecar_live")
    if peak_cuda_gib > device_budget_gib:
        blockers.append("representative_map_peak_cuda_exceeds_device_budget")

    return {
        "probe": "t038_representative_map_gate",
        "activation": REPRESENTATIVE_MAP_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": {
            "tokens_per_second": TARGET_TOKENS_PER_SECOND,
            "seconds_per_token": TARGET_SECONDS_PER_TOKEN,
        },
        "baseline": {
            "source": "R452/T028 direct denominator",
            "kernel_seconds_per_token": R452_KERNEL_SECONDS_PER_TOKEN,
            "required_kernel_speedup_if_control_zero": required_speedup,
            "current_best_reference": "T017 synthetic MLP-like packed-linear gate",
            "current_best_reference_speedup": T017_MLP_LIKE_SYNTHETIC_SPEEDUP,
        },
        "representative_map": {
            "representative_speedup_vs_current": representative_speedup_vs_current,
            "projected_kernel_seconds_per_token": projected_kernel_seconds,
            "projected_direct_seconds_if_control_zero": projected_direct_seconds_if_control_zero,
            "projected_tokens_per_second_if_control_zero": projected_tokens_per_second,
            "covers_full_body_fraction": covers_full_body_fraction,
            "exact_output_equivalence": exact_output_equivalence,
            "hidden_sidecar_gib": hidden_sidecar_gib,
            "peak_cuda_gib": peak_cuda_gib,
            "device_budget_gib": device_budget_gib,
        },
        "readiness": {
            "t038_representative_map_gate_ready": gate_ready,
            "t038_representative_map_speedup_sufficient": speed_pass,
            "t038_representative_map_exact": exact_output_equivalence,
            "t038_representative_map_covers_body": coverage_pass,
            "t038_representative_map_no_hidden_sidecar": hidden_sidecar_gib == 0.0,
            "t038_representative_map_fits_device_budget": peak_cuda_gib <= device_budget_gib,
            "t038_production_implementation_allowed": False,
        },
        "blockers": blockers,
        "next_actions": []
        if gate_ready
        else [
            "Do not build a whole-token VM on the current representative map kernel.",
            "Research or implement a new exact map body that exceeds 7.5738x before VM integration.",
        ],
    }


def run_t038b_coverage_denominator_gate(
    *,
    component_names: tuple[str, ...] = ("exact_compact_body", "unchanged_leftover_body"),
    component_fractions: tuple[float, ...] = (0.80, 0.20),
    component_speedups: tuple[float, ...] = (
        R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN,
        1.0,
    ),
    minimum_accelerated_body_fraction: float = 0.80,
    fraction_tolerance: float = 1e-6,
    hidden_sidecar_gib: float = 0.0,
    decoded_dense_sidecar_gib: float = 0.0,
    residual_fraction: float = 0.0,
    exact_arithmetic: bool = True,
    exact_layout_semantics: bool = True,
    layout_replaces_storage: bool = True,
    peak_cuda_gib: float = R452_PEAK_CUDA_GIB,
    device_budget_gib: float = DEFAULT_DEVICE_BUDGET_GIB,
) -> dict[str, Any]:
    """Price a composed exact body-map replacement without hidden leftovers."""

    input_errors = _validate_component_inputs(
        component_names=component_names,
        component_fractions=component_fractions,
        component_speedups=component_speedups,
    )
    required_speedup = R452_KERNEL_SECONDS_PER_TOKEN / TARGET_SECONDS_PER_TOKEN
    fraction_sum = sum(component_fractions)
    unpriced_leftover_fraction = max(0.0, 1.0 - fraction_sum)
    fraction_sum_pass = abs(fraction_sum - 1.0) <= fraction_tolerance
    accelerated_body_fraction = sum(
        fraction for fraction, speedup in zip(component_fractions, component_speedups, strict=False) if speedup > 1.0
    )
    coverage_pass = accelerated_body_fraction >= minimum_accelerated_body_fraction
    harmonic_denominator = sum(
        fraction / max(speedup, 1e-12)
        for fraction, speedup in zip(component_fractions, component_speedups, strict=False)
    )
    composite_speedup = 1.0 / harmonic_denominator if harmonic_denominator > 0.0 and not input_errors else 0.0
    projected_kernel_seconds = (
        R452_KERNEL_SECONDS_PER_TOKEN / composite_speedup if composite_speedup > 0.0 else float("inf")
    )
    projected_tokens_per_second_if_control_zero = (
        1.0 / projected_kernel_seconds if projected_kernel_seconds > 0.0 else 0.0
    )
    sidecar_gib = hidden_sidecar_gib + decoded_dense_sidecar_gib
    residual_zero = residual_fraction == 0.0
    sidecar_pass = sidecar_gib == 0.0
    device_pass = peak_cuda_gib <= device_budget_gib
    composite_speedup_pass = composite_speedup >= required_speedup
    gate_ready = (
        not input_errors
        and fraction_sum_pass
        and unpriced_leftover_fraction <= fraction_tolerance
        and coverage_pass
        and composite_speedup_pass
        and sidecar_pass
        and residual_zero
        and exact_arithmetic
        and exact_layout_semantics
        and layout_replaces_storage
        and device_pass
    )

    components: list[dict[str, Any]] = []
    for name, fraction, speedup in zip(component_names, component_fractions, component_speedups, strict=False):
        contribution = fraction / speedup if speedup > 0.0 else float("inf")
        components.append(
            {
                "name": name,
                "body_fraction": fraction,
                "speedup_vs_current_component": speedup,
                "harmonic_denominator_contribution": contribution,
                "accelerated": speedup > 1.0,
            }
        )

    blockers: list[str] = list(input_errors)
    if not fraction_sum_pass:
        blockers.append("body_component_fractions_do_not_sum_to_one")
    if unpriced_leftover_fraction > fraction_tolerance:
        blockers.append("unpriced_leftover_body_fraction")
    if not coverage_pass:
        blockers.append("accelerated_body_fraction_below_required_80_percent")
    if not composite_speedup_pass:
        blockers.append("composite_body_speedup_below_required_7_5738x")
    if not sidecar_pass:
        blockers.append("hidden_or_decoded_dense_sidecar_live")
    if not residual_zero:
        blockers.append("exact_map_residual_is_nonzero")
    if not exact_arithmetic:
        blockers.append("exact_arithmetic_not_proven")
    if not exact_layout_semantics:
        blockers.append("exact_layout_semantics_not_proven")
    if not layout_replaces_storage:
        blockers.append("layout_or_atom_program_does_not_replace_storage")
    if not device_pass:
        blockers.append("projected_peak_cuda_exceeds_device_budget")

    return {
        "probe": "t038b_coverage_denominator_gate",
        "activation": T038B_COVERAGE_ACTIVATION,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": {
            "tokens_per_second": TARGET_TOKENS_PER_SECOND,
            "seconds_per_token": TARGET_SECONDS_PER_TOKEN,
            "required_kernel_speedup_if_control_zero": required_speedup,
        },
        "baseline": {
            "source": "R452/T028 direct denominator",
            "kernel_seconds_per_token": R452_KERNEL_SECONDS_PER_TOKEN,
            "current_tokens_per_second": R452_TOKENS_PER_SECOND,
        },
        "coverage_model": {
            "component_fraction_sum": fraction_sum,
            "unpriced_leftover_fraction": unpriced_leftover_fraction,
            "accelerated_body_fraction": accelerated_body_fraction,
            "minimum_accelerated_body_fraction": minimum_accelerated_body_fraction,
            "harmonic_denominator": harmonic_denominator,
            "composite_speedup_vs_current_body": composite_speedup,
            "projected_kernel_seconds_per_token": projected_kernel_seconds,
            "projected_tokens_per_second_if_control_zero": projected_tokens_per_second_if_control_zero,
        },
        "components": components,
        "exactness_audit": {
            "hidden_sidecar_gib": hidden_sidecar_gib,
            "decoded_dense_sidecar_gib": decoded_dense_sidecar_gib,
            "sidecar_gib": sidecar_gib,
            "residual_fraction": residual_fraction,
            "exact_arithmetic": exact_arithmetic,
            "exact_layout_semantics": exact_layout_semantics,
            "layout_replaces_storage": layout_replaces_storage,
            "peak_cuda_gib": peak_cuda_gib,
            "device_budget_gib": device_budget_gib,
        },
        "readiness": {
            "t038b_coverage_denominator_gate_ready": gate_ready,
            "t038b_component_fractions_sum_to_one": fraction_sum_pass,
            "t038b_no_unpriced_leftover": unpriced_leftover_fraction <= fraction_tolerance,
            "t038b_accelerated_body_fraction_sufficient": coverage_pass,
            "t038b_composite_speedup_sufficient": composite_speedup_pass,
            "t038b_no_hidden_sidecar": sidecar_pass,
            "t038b_zero_residual": residual_zero,
            "t038b_exact_arithmetic": exact_arithmetic,
            "t038b_exact_layout_semantics": exact_layout_semantics,
            "t038b_layout_replaces_storage": layout_replaces_storage,
            "t038b_fits_device_budget": device_pass,
            "t038b_production_implementation_allowed": False,
        },
        "blockers": blockers,
        "next_actions": [
            "Promote the surviving layout or atom object to a representative exact map implementation path.",
            "Measure the production GPU body path before claiming a real token/s gain.",
        ]
        if gate_ready
        else [
            "Do not implement a production body replacement until every body fraction is priced and the composed denominator crosses 7.5738x.",
            "If the strongest layout/atom candidates cannot satisfy this gate, run a new research wave focused on exact map objects rather than verifier or control-path work.",
        ],
    }


__all__ = [
    "ACTIVATION",
    "REPRESENTATIVE_MAP_ACTIVATION",
    "T038B_COVERAGE_ACTIVATION",
    "run_t038_kernel_body_roofline_gate",
    "run_t038_representative_map_gate",
    "run_t038b_coverage_denominator_gate",
]
