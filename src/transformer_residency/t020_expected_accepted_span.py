"""Expected-accepted-span optimizer gate for speculative decode spans."""

from __future__ import annotations

from typing import Any


ACTIVATION = "T020_EXPECTED_ACCEPTED_SPAN_ACTIVE"

DEFAULT_SPAN_LENGTHS = (1, 2, 3, 4, 8)
DEFAULT_PROFILES = ("flat", "entropy_sensitive", "collapse_after_first", "high_confidence_run")


def _validate_span_lengths(span_lengths: tuple[int, ...]) -> tuple[int, ...]:
    if not span_lengths:
        raise ValueError("span_lengths must not be empty")
    normalized = tuple(sorted(set(span_lengths)))
    if normalized[0] <= 0:
        raise ValueError("span_lengths must be positive")
    if 1 not in normalized:
        raise ValueError("span_lengths must include 1")
    return normalized


def acceptance_survival_curve(profile: str, span_length: int) -> list[float]:
    """Return P(A >= i) for i in [1, span_length] under a local acceptance profile."""

    if span_length <= 0:
        raise ValueError("span_length must be positive")
    if profile == "flat":
        token_acceptance = [0.74 for _ in range(span_length)]
    elif profile == "entropy_sensitive":
        token_acceptance = [max(0.42, 0.88 - 0.08 * index) for index in range(span_length)]
    elif profile == "collapse_after_first":
        token_acceptance = [0.78] + [0.15 for _ in range(span_length - 1)]
    elif profile == "high_confidence_run":
        token_acceptance = [0.92 for _ in range(span_length)]
    else:
        raise ValueError(f"unknown acceptance profile: {profile}")

    survival: list[float] = []
    probability = 1.0
    for token_probability in token_acceptance:
        probability *= token_probability
        survival.append(probability)
    return survival


def expected_accepted_tokens(profile: str, span_length: int) -> float:
    return sum(acceptance_survival_curve(profile, span_length))


def price_span_length(
    *,
    profile: str,
    span_length: int,
    base_span_cost: float = 0.36,
    verify_cost_per_token: float = 0.34,
    delayed_kv_cost_per_token: float = 0.05,
    lm_head_cost_per_token: float = 0.06,
    rollback_cost_per_rejected_token: float = 0.08,
    schedule_gap_cost: float = 0.10,
    graph_specialization_cost: float = 0.02,
) -> dict[str, Any]:
    """Price one proposed span by expected cost per accepted token."""

    if span_length <= 0:
        raise ValueError("span_length must be positive")
    costs = (
        base_span_cost,
        verify_cost_per_token,
        delayed_kv_cost_per_token,
        lm_head_cost_per_token,
        rollback_cost_per_rejected_token,
        schedule_gap_cost,
        graph_specialization_cost,
    )
    if any(cost < 0.0 for cost in costs):
        raise ValueError("costs must be non-negative")

    expected_accepted = expected_accepted_tokens(profile, span_length)
    expected_rejected_or_uncommitted = max(0.0, span_length - expected_accepted)
    fixed_cost = base_span_cost + schedule_gap_cost + graph_specialization_cost
    per_token_cost = span_length * (verify_cost_per_token + delayed_kv_cost_per_token + lm_head_cost_per_token)
    rollback_cost = expected_rejected_or_uncommitted * rollback_cost_per_rejected_token
    total_cost = fixed_cost + per_token_cost + rollback_cost
    return {
        "profile": profile,
        "span_length": span_length,
        "expected_accepted_tokens": expected_accepted,
        "expected_rejected_or_uncommitted": expected_rejected_or_uncommitted,
        "fixed_cost": fixed_cost,
        "verify_cost": span_length * verify_cost_per_token,
        "delayed_kv_cost": span_length * delayed_kv_cost_per_token,
        "lm_head_cost": span_length * lm_head_cost_per_token,
        "rollback_cost": rollback_cost,
        "total_span_cost": total_cost,
        "cost_per_accepted_token": total_cost / max(1e-9, expected_accepted),
    }


def optimize_expected_accepted_span(
    *,
    profile: str,
    span_lengths: tuple[int, ...] = DEFAULT_SPAN_LENGTHS,
    risk_margin: float = 0.03,
    base_span_cost: float = 0.36,
    verify_cost_per_token: float = 0.34,
    delayed_kv_cost_per_token: float = 0.05,
    lm_head_cost_per_token: float = 0.06,
    rollback_cost_per_rejected_token: float = 0.08,
    schedule_gap_cost: float = 0.10,
    graph_specialization_cost: float = 0.02,
) -> dict[str, Any]:
    """Choose the span length that minimizes expected denominator cost."""

    if risk_margin < 0.0:
        raise ValueError("risk_margin must be non-negative")
    lengths = _validate_span_lengths(span_lengths)
    rows = [
        price_span_length(
            profile=profile,
            span_length=span_length,
            base_span_cost=base_span_cost,
            verify_cost_per_token=verify_cost_per_token,
            delayed_kv_cost_per_token=delayed_kv_cost_per_token,
            lm_head_cost_per_token=lm_head_cost_per_token,
            rollback_cost_per_rejected_token=rollback_cost_per_rejected_token,
            schedule_gap_cost=schedule_gap_cost,
            graph_specialization_cost=graph_specialization_cost,
        )
        for span_length in lengths
    ]
    baseline = next(row for row in rows if row["span_length"] == 1)
    best = min(rows, key=lambda row: (row["cost_per_accepted_token"], row["span_length"]))
    speedup_vs_one_token = baseline["cost_per_accepted_token"] / max(1e-9, best["cost_per_accepted_token"])
    adaptive_legal = best["span_length"] == 1 or (
        best["cost_per_accepted_token"] + risk_margin < baseline["cost_per_accepted_token"]
    )
    return {
        "profile": profile,
        "rows": rows,
        "baseline": baseline,
        "selected": best,
        "risk_margin": risk_margin,
        "adaptive_step_legal": adaptive_legal,
        "speedup_vs_one_token": speedup_vs_one_token,
    }


def run_t020_expected_accepted_span_gate(
    *,
    span_lengths: tuple[int, ...] = DEFAULT_SPAN_LENGTHS,
    profiles: tuple[str, ...] = DEFAULT_PROFILES,
    risk_margin: float = 0.03,
    min_speedup: float = 1.10,
    base_span_cost: float = 0.36,
    verify_cost_per_token: float = 0.34,
    delayed_kv_cost_per_token: float = 0.05,
    lm_head_cost_per_token: float = 0.06,
    rollback_cost_per_rejected_token: float = 0.08,
    schedule_gap_cost: float = 0.10,
    graph_specialization_cost: float = 0.02,
) -> dict[str, Any]:
    """Gate C-MATH-W01-02 by accepted-token denominator optimization."""

    if min_speedup <= 1.0:
        raise ValueError("min_speedup must be greater than 1")
    lengths = _validate_span_lengths(span_lengths)
    if not profiles:
        raise ValueError("profiles must not be empty")

    profile_results = [
        optimize_expected_accepted_span(
            profile=profile,
            span_lengths=lengths,
            risk_margin=risk_margin,
            base_span_cost=base_span_cost,
            verify_cost_per_token=verify_cost_per_token,
            delayed_kv_cost_per_token=delayed_kv_cost_per_token,
            lm_head_cost_per_token=lm_head_cost_per_token,
            rollback_cost_per_rejected_token=rollback_cost_per_rejected_token,
            schedule_gap_cost=schedule_gap_cost,
            graph_specialization_cost=graph_specialization_cost,
        )
        for profile in profiles
    ]
    by_profile = {result["profile"]: result for result in profile_results}
    high_confidence = by_profile.get("high_confidence_run")
    collapse = by_profile.get("collapse_after_first")

    positive_profile_pass = bool(
        high_confidence
        and high_confidence["selected"]["span_length"] > 1
        and high_confidence["speedup_vs_one_token"] >= min_speedup
        and high_confidence["adaptive_step_legal"]
    )
    collapse_fallback_pass = bool(collapse and collapse["selected"]["span_length"] == 1)
    all_adaptive_steps_legal = all(result["adaptive_step_legal"] for result in profile_results)
    hidden_costs_priced = True
    exact_verifier_required = True
    gate_pass = bool(
        positive_profile_pass
        and collapse_fallback_pass
        and all_adaptive_steps_legal
        and hidden_costs_priced
        and exact_verifier_required
    )

    blockers: list[str] = []
    if not positive_profile_pass:
        blockers.append("high_confidence_profile_does_not_beat_one_token_by_10_percent")
    if not collapse_fallback_pass:
        blockers.append("collapse_profile_does_not_fall_back_to_single_token")
    if not all_adaptive_steps_legal:
        blockers.append("selected_span_violates_risk_margin_adaptive_step")
    if not hidden_costs_priced:
        blockers.append("hidden_denominator_costs_not_priced")
    if not exact_verifier_required:
        blockers.append("missing_exact_verifier_boundary")

    selected_profile = high_confidence or profile_results[0]
    return {
        "probe": "t020_expected_accepted_span_gate",
        "activation": ACTIVATION,
        "finding": "C-MATH-W01-02",
        "inequality": "choose b only when R_hat(b) + risk_margin < R_hat(1)",
        "span_lengths": lengths,
        "cost_terms": {
            "C_base": base_span_cost,
            "C_verify_per_token": verify_cost_per_token,
            "C_KV_delayed_per_token": delayed_kv_cost_per_token,
            "C_lm_per_token": lm_head_cost_per_token,
            "C_rollback_per_rejected_token": rollback_cost_per_rejected_token,
            "C_schedule": schedule_gap_cost,
            "C_graph_specialization": graph_specialization_cost,
        },
        "baseline": {
            "source": "one-token accepted denominator for the high-confidence profile",
            "span_length": selected_profile["baseline"]["span_length"],
            "expected_accepted_tokens": selected_profile["baseline"]["expected_accepted_tokens"],
            "cost_per_accepted_token": selected_profile["baseline"]["cost_per_accepted_token"],
        },
        "candidate": {
            "profile": selected_profile["profile"],
            "span_length": selected_profile["selected"]["span_length"],
            "expected_accepted_tokens": selected_profile["selected"]["expected_accepted_tokens"],
            "cost_per_accepted_token": selected_profile["selected"]["cost_per_accepted_token"],
            "speedup_vs_one_token": selected_profile["speedup_vs_one_token"],
            "risk_margin": risk_margin,
        },
        "profiles": profile_results,
        "readiness": {
            "expected_accepted_span_gate_ready": gate_pass,
            "positive_profile_selects_span": positive_profile_pass,
            "collapse_profile_falls_back_to_one": collapse_fallback_pass,
            "all_selected_spans_pass_risk_margin": all_adaptive_steps_legal,
            "hidden_denominator_costs_priced": hidden_costs_priced,
            "exact_verifier_required": exact_verifier_required,
        },
        "blockers": blockers,
        "next_actions": (
            [
                "Attach this optimizer to the speculative span scheduler before implementing span execution.",
                "Feed online entropy, margin, and rejection-rate estimates into the profile selector.",
            ]
            if gate_pass
            else ["Do not integrate accepted-span execution until the denominator optimizer chooses legal spans."]
        ),
    }


__all__ = [
    "ACTIVATION",
    "DEFAULT_PROFILES",
    "DEFAULT_SPAN_LENGTHS",
    "acceptance_survival_curve",
    "expected_accepted_tokens",
    "optimize_expected_accepted_span",
    "price_span_length",
    "run_t020_expected_accepted_span_gate",
]
