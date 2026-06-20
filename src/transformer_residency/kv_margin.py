"""KV-cache margin gates for the transformer residency line."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .footprint import BYTES_PER_GIB, TransformerShape, qwen3_32b_shape


@dataclass(frozen=True)
class KVMarginCandidate:
    id: str
    aspect: str
    name: str
    strength: str
    projected_gain: str
    gate_strength: str
    tradeoff: str
    near_no_tradeoff: bool

    @property
    def is_strong(self) -> bool:
        return self.strength == "strong"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class KVMarginScenario:
    name: str
    context_tokens: int
    weight_gib: float
    kv_gib: float
    total_gib: float
    slack_gib: float
    synthetic_tokens_per_second: float
    speedup_vs_baseline: float
    vram_saved_vs_baseline_gib: float
    notes: str

    @property
    def fits_12gib(self) -> bool:
        return self.total_gib <= 12.0

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["fits_12gib"] = self.fits_12gib
        return data


def qwen3_32b_kv_bytes_per_token(shape: TransformerShape | None = None) -> int:
    """Return full FP16 K+V cache bytes per token for the Qwen3-32B shape."""

    current_shape = shape or qwen3_32b_shape()
    return current_shape.layers * 2 * current_shape.kv_heads * current_shape.head_dim * 2


def kv_cache_gib_for_context(
    context_tokens: int,
    *,
    effective_bits: float = 16.0,
    shape: TransformerShape | None = None,
    metadata_overhead_gib: float = 0.0,
) -> float:
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if effective_bits <= 0.0:
        raise ValueError("effective_bits must be positive")
    fp16_bytes = qwen3_32b_kv_bytes_per_token(shape)
    return (context_tokens * fp16_bytes * (effective_bits / 16.0)) / BYTES_PER_GIB + metadata_overhead_gib


def _weight_total_gib(
    shape: TransformerShape,
    *,
    mlp_bits: float,
    attention_bits: float,
    embedding_bits: float,
    output_head_bits: float,
    overhead_gib: float,
) -> float:
    return float(
        shape.component_residency_projection(
            mlp_bits=mlp_bits,
            attention_bits=attention_bits,
            embedding_bits=embedding_bits,
            output_head_bits=output_head_bits,
            overhead_gib=overhead_gib,
        )["candidate_estimated_gib"]
    )


def _synthetic_decode_tokens_per_second(
    shape: TransformerShape,
    *,
    mlp_bits: float,
    attention_bits: float,
    embedding_bits: float,
    output_head_bits: float,
    context_tokens: int,
    kv_effective_bits: float,
    bandwidth_gib_per_second: float,
    exact_speed_multiplier: float = 1.0,
) -> float:
    if exact_speed_multiplier <= 0.0:
        raise ValueError("exact_speed_multiplier must be positive")
    throughput = shape.synthetic_decode_throughput_projection(
        uniform_bits=3.0,
        mlp_bits=mlp_bits,
        attention_bits=attention_bits,
        embedding_bits=embedding_bits,
        output_head_bits=output_head_bits,
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        kv_cache_gib_per_token=kv_cache_gib_for_context(context_tokens, effective_bits=kv_effective_bits, shape=shape),
    )
    return float(throughput["component_tokens_per_second"]) * exact_speed_multiplier


def _scenario(
    shape: TransformerShape,
    *,
    name: str,
    context_tokens: int,
    mlp_bits: float,
    attention_bits: float,
    embedding_bits: float,
    output_head_bits: float,
    overhead_gib: float,
    kv_effective_bits: float,
    baseline_total_gib: float,
    baseline_tokens_per_second: float,
    bandwidth_gib_per_second: float,
    exact_speed_multiplier: float = 1.0,
    metadata_overhead_gib: float = 0.0,
    notes: str,
) -> KVMarginScenario:
    weight_gib = _weight_total_gib(
        shape,
        mlp_bits=mlp_bits,
        attention_bits=attention_bits,
        embedding_bits=embedding_bits,
        output_head_bits=output_head_bits,
        overhead_gib=overhead_gib,
    )
    kv_gib = kv_cache_gib_for_context(
        context_tokens,
        effective_bits=kv_effective_bits,
        shape=shape,
        metadata_overhead_gib=metadata_overhead_gib,
    )
    total_gib = weight_gib + kv_gib
    tok_s = _synthetic_decode_tokens_per_second(
        shape,
        mlp_bits=mlp_bits,
        attention_bits=attention_bits,
        embedding_bits=embedding_bits,
        output_head_bits=output_head_bits,
        context_tokens=context_tokens,
        kv_effective_bits=kv_effective_bits,
        bandwidth_gib_per_second=bandwidth_gib_per_second,
        exact_speed_multiplier=exact_speed_multiplier,
    )
    return KVMarginScenario(
        name=name,
        context_tokens=context_tokens,
        weight_gib=weight_gib,
        kv_gib=kv_gib,
        total_gib=total_gib,
        slack_gib=12.0 - total_gib,
        synthetic_tokens_per_second=tok_s,
        speedup_vs_baseline=tok_s / baseline_tokens_per_second,
        vram_saved_vs_baseline_gib=baseline_total_gib - total_gib,
        notes=notes,
    )


def t002_candidates() -> tuple[KVMarginCandidate, ...]:
    rows = (
        ("T002-01", "A1", "Recent-FP16 plus old-INT8 KV with per-head residual bound", "strong", "8k KV 2.0 -> ~1.0 GiB", "high", "quant kernels and scale metadata", True),
        ("T002-02", "A1", "KIVI-style per-channel K / per-token V at 2-4 bits with recent exemption", "strong", "16k KV 4.0 -> 0.5-1.0 GiB", "high", "quality debt without residual gate", True),
        ("T002-03", "A1", "Pre-RoPE key quantization with phase reconstruction", "strong", "safer 4-bit keys", "medium_high", "extra phase handling", True),
        ("T002-04", "A1", "Layer/head adaptive KV bit allocation", "strong", "20-40% over uniform low-bit KV", "high", "calibration pass", True),
        ("T002-05", "A1", "Value-cache dense+sparse outlier lanes", "strong", "V cache 2-4x smaller", "medium_high", "sparse lane dispatch", True),
        ("T002-06", "A1", "QJL/sign residual correction for low-bit key dot-products", "strong", "3-4 bit keys act higher precision", "medium", "kernel complexity", True),
        ("T002-07", "A2", "Exact-recent plus attention-sink anchors", "strong", "large streaming-context memory cut", "high", "retrieval-specific gate", True),
        ("T002-08", "A2", "Pyramid layer budget", "strong", "retain 12-25% long-context KV", "medium_high", "task-dependent", False),
        ("T002-09", "A2", "Heavy-hitter protected retention", "strong", "2-5x long-context KV reduction", "medium_high", "score overhead", False),
        ("T002-10", "A2", "Semantic pinning for delimiters, code anchors, tool calls, retrieval spans", "strong", "lower retrieval risk at same budget", "medium", "cheap classifier required", True),
        ("T002-11", "A2", "Prefix KV copy-on-write and page sharing", "strong", "1.1-3x effective memory on repeated prefixes", "high", "workload-dependent", True),
        ("T002-12", "A3", "Activation-informed KV-head folding from 8 to 4 heads", "strong", "2x KV head denominator", "medium", "calibration or small fine-tune", False),
        ("T002-13", "A3", "MLA retrofit / MHA2MLA latent KV adapter", "medium", "3-10x KV possible", "medium", "architecture change", False),
        ("T002-14", "A3", "Cross-layer KV subspace sharing", "strong", "2-3x KV compression", "medium", "layer-specific task risk", False),
        ("T002-15", "A3", "Low-rank KV adaptation inside GQA groups", "strong", "2-4x KV cache reduction", "medium", "adapter training", False),
        ("T002-16", "A3", "Layer-paired K reuse in redundant middle layers", "medium", "1.2-1.8x KV reduction", "low_medium", "higher quality risk", False),
        ("T002-17", "A4", "T001 MLP quotient total-bit tightening", "strong", "0.5-0.7 GiB weight margin", "high", "captured tile proof", True),
        ("T002-18", "A4", "Q/K/V/O asymmetric precision lanes", "strong", "0.15-0.35 GiB margin", "medium_high", "attention projection fragility", True),
        ("T002-19", "A4", "Embedding/lm-head hot-row mixed precision", "strong", "0.15-0.45 GiB margin", "medium_high", "rare-token quality gate", True),
        ("T002-20", "A4", "Scale/metadata superblock amortization", "strong", "0.10-0.25 GiB margin", "high", "engineering-only", True),
        ("T002-21", "A4", "Static runtime arena", "strong", "0.45 GiB immediate slack", "high", "platform-specific", True),
        ("T002-22", "A4", "Decode scratch reuse and activation aliasing", "strong", "0.10-0.30 GiB slack", "high", "live-range proof", True),
        ("T002-23", "A5", "Self-speculative draft/verify from same model path", "strong", "1.2-1.8x tok/s", "high", "draft path", True),
        ("T002-24", "A5", "Verified lm-head shortlist with exact fallback", "strong", "1.1-1.4x tok/s", "high", "shortlist predictor", True),
        ("T002-25", "A5", "Quantized-KV FlashAttention/paged-decode fused kernel", "strong", "1.2-2.0x attention-bound tok/s", "high", "kernel work", True),
        ("T002-26", "A5", "CUDA graph / persistent decode loop", "medium", "1.05-1.25x tok/s", "high", "product support", True),
        ("T002-27", "A6", "Fixed-size KV block allocator", "strong", "0.1-0.4 GiB slack", "high", "allocator support", True),
        ("T002-28", "A6", "Cold-context on-GPU tier", "strong", "16k with ~1 GiB KV", "high", "page metadata", True),
        ("T002-29", "A6", "KV/weight co-packing reserves KV before optional repair atoms", "strong", "protects KV margin", "high", "lower optional repair budget", True),
        ("T002-30", "A6", "Bandwidth-aware layout for scales, packed values, attention reads", "medium", "1.05-1.20x tok/s", "medium_high", "platform-specific", True),
        ("T002-31", "A7", "KV-visible quotient certificate", "strong", "2-8x KV reduction for certified tokens", "high", "careful falsifier", True),
        ("T002-32", "A7", "Joint weight-KV residual budget allocator", "strong", "0.5-1.5 GiB stack margin", "high", "new control policy", True),
    )
    return tuple(KVMarginCandidate(*row) for row in rows)


def run_t002_kv_margin_sweep(
    *,
    bandwidth_gib_per_second: float = 500.0,
    shape: TransformerShape | None = None,
) -> dict[str, object]:
    current_shape = shape or qwen3_32b_shape()
    baseline_weight_gib = _weight_total_gib(
        current_shape,
        mlp_bits=2.65,
        attention_bits=3.50,
        embedding_bits=4.0,
        output_head_bits=4.0,
        overhead_gib=1.0,
    )
    baseline_kv_gib = kv_cache_gib_for_context(4096, effective_bits=16.0, shape=current_shape)
    baseline_total_gib = baseline_weight_gib + baseline_kv_gib
    baseline_tokens_per_second = _synthetic_decode_tokens_per_second(
        current_shape,
        mlp_bits=2.65,
        attention_bits=3.50,
        embedding_bits=4.0,
        output_head_bits=4.0,
        context_tokens=4096,
        kv_effective_bits=16.0,
        bandwidth_gib_per_second=bandwidth_gib_per_second,
    )

    scenarios = (
        KVMarginScenario(
            name="before_t001_component_fp16_kv_4k",
            context_tokens=4096,
            weight_gib=baseline_weight_gib,
            kv_gib=baseline_kv_gib,
            total_gib=baseline_total_gib,
            slack_gib=12.0 - baseline_total_gib,
            synthetic_tokens_per_second=baseline_tokens_per_second,
            speedup_vs_baseline=1.0,
            vram_saved_vs_baseline_gib=0.0,
            notes="Current T001 component stack plus ordinary FP16 KV.",
        ),
        _scenario(
            current_shape,
            name="minimal_static_arena_light_tighten_int8_kv_8k",
            context_tokens=8192,
            mlp_bits=2.38,
            attention_bits=3.35,
            embedding_bits=3.75,
            output_head_bits=3.75,
            overhead_gib=0.55,
            kv_effective_bits=8.0,
            baseline_total_gib=baseline_total_gib,
            baseline_tokens_per_second=baseline_tokens_per_second,
            bandwidth_gib_per_second=bandwidth_gib_per_second,
            notes="T002-01/T002-17/T002-21; doubles usable context after correcting the Qwen3 head_dim denominator.",
        ),
        _scenario(
            current_shape,
            name="strong_weight_tighten_tiered_int4_kv_16k",
            context_tokens=16384,
            mlp_bits=2.20,
            attention_bits=3.00,
            embedding_bits=3.50,
            output_head_bits=3.50,
            overhead_gib=0.55,
            kv_effective_bits=4.0,
            baseline_total_gib=baseline_total_gib,
            baseline_tokens_per_second=baseline_tokens_per_second,
            bandwidth_gib_per_second=bandwidth_gib_per_second,
            notes="T002-02/T002-17/T002-21/T002-28; near-term target with almost one GiB slack.",
        ),
        _scenario(
            current_shape,
            name="aggressive_quotient_int2_kv_32k",
            context_tokens=32768,
            mlp_bits=2.20,
            attention_bits=3.00,
            embedding_bits=3.50,
            output_head_bits=3.50,
            overhead_gib=0.55,
            kv_effective_bits=2.0,
            metadata_overhead_gib=0.10,
            baseline_total_gib=baseline_total_gib,
            baseline_tokens_per_second=baseline_tokens_per_second,
            bandwidth_gib_per_second=bandwidth_gib_per_second,
            notes="T002-31/T002-32 aggressive certificate path; high upside, stricter quality gate.",
        ),
        _scenario(
            current_shape,
            name="tok_s_push_exact_self_spec_int4_16k",
            context_tokens=16384,
            mlp_bits=2.20,
            attention_bits=3.00,
            embedding_bits=3.50,
            output_head_bits=3.50,
            overhead_gib=0.55,
            kv_effective_bits=4.0,
            baseline_total_gib=baseline_total_gib,
            baseline_tokens_per_second=baseline_tokens_per_second,
            bandwidth_gib_per_second=bandwidth_gib_per_second,
            exact_speed_multiplier=1.35,
            notes="T002-23 exact self-speculative verifier layered on the strong 16k memory stack.",
        ),
    )
    candidates = t002_candidates()
    strong_count = sum(1 for candidate in candidates if candidate.is_strong)
    near_no_tradeoff_strong_count = sum(
        1 for candidate in candidates if candidate.is_strong and candidate.near_no_tradeoff
    )
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    validations = {
        "activation": "T002_KV_MARGIN_SWEEP_ACTIVE",
        "candidate_count_at_least_30": len(candidates) >= 30,
        "strong_count_at_least_20": strong_count >= 20,
        "near_no_tradeoff_strong_count_at_least_20": near_no_tradeoff_strong_count >= 20,
        "qwen3_fp16_kv_bytes_per_token_is_262144": qwen3_32b_kv_bytes_per_token(current_shape) == 262_144,
        "baseline_fp16_4k_is_fragile": scenario_by_name["before_t001_component_fp16_kv_4k"].slack_gib < 0.05,
        "minimal_8k_has_positive_margin": scenario_by_name[
            "minimal_static_arena_light_tighten_int8_kv_8k"
        ].slack_gib
        >= 0.35,
        "strong_16k_has_positive_margin": scenario_by_name["strong_weight_tighten_tiered_int4_kv_16k"].slack_gib >= 0.75,
        "aggressive_32k_has_positive_margin": scenario_by_name["aggressive_quotient_int2_kv_32k"].slack_gib >= 0.60,
        "tok_s_push_improves_synthetic_decode": scenario_by_name[
            "tok_s_push_exact_self_spec_int4_16k"
        ].speedup_vs_baseline
        >= 1.20,
    }
    return {
        "probe": "t002_kv_margin_sweep",
        "activation": "T002_KV_MARGIN_SWEEP_ACTIVE",
        "shape": current_shape.as_dict(),
        "kv_denominator": {
            "fp16_bytes_per_token": qwen3_32b_kv_bytes_per_token(current_shape),
            "fp16_kib_per_token": qwen3_32b_kv_bytes_per_token(current_shape) / 1024.0,
            "fp16_gib_by_context": {
                str(context): kv_cache_gib_for_context(context, effective_bits=16.0, shape=current_shape)
                for context in (4096, 8192, 16384, 32768)
            },
        },
        "candidate_summary": {
            "total": len(candidates),
            "strong": strong_count,
            "near_no_tradeoff_strong": near_no_tradeoff_strong_count,
            "medium": sum(1 for candidate in candidates if candidate.strength == "medium"),
        },
        "candidates": [candidate.as_dict() for candidate in candidates],
        "scenarios": [scenario.as_dict() for scenario in scenarios],
        "validations": validations,
        "overall": {
            "pass": all(bool(value) for value in validations.values() if isinstance(value, bool)),
            "before_total_gib": scenario_by_name["before_t001_component_fp16_kv_4k"].total_gib,
            "best_margin_total_gib": scenario_by_name["strong_weight_tighten_tiered_int4_kv_16k"].total_gib,
            "best_margin_slack_gib": scenario_by_name["strong_weight_tighten_tiered_int4_kv_16k"].slack_gib,
            "best_context_tokens_with_margin": 32768
            if scenario_by_name["aggressive_quotient_int2_kv_32k"].fits_12gib
            else 16384,
            "best_tok_s_speedup": scenario_by_name["tok_s_push_exact_self_spec_int4_16k"].speedup_vs_baseline,
        },
    }
