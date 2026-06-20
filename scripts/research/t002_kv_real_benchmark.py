"""Run a local CUDA benchmark for the T002 KV-margin projection.

This benchmark intentionally does not download or run Qwen3-32B. It validates
the denominator with real CUDA allocations and measures local GPU memory
bandwidth, then prices the T002 scenarios with that measured bandwidth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_residency import run_t002_kv_margin_sweep

BYTES_PER_GIB = 1024**3
DEFAULT_OUT = ROOT / ".work" / "jobs" / "t002-kv-real-benchmark"


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("PyTorch is required for the CUDA benchmark") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    return torch


def _measure_cuda_copy_bandwidth(torch, gib: float, repeats: int, warmup: int) -> dict[str, float | int | str]:
    if gib <= 0.0:
        raise ValueError("gib must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    device = torch.device("cuda")
    elements = int(gib * BYTES_PER_GIB)
    source = torch.empty(elements, dtype=torch.uint8, device=device)
    target = torch.empty_like(source)
    source.fill_(17)
    torch.cuda.synchronize()

    for _ in range(warmup):
        target.copy_(source)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        target.copy_(source)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = float(start.elapsed_time(end))
    copied_gib = gib * repeats
    traffic_gib = copied_gib * 2.0
    seconds = elapsed_ms / 1000.0
    bandwidth_gib_s = traffic_gib / seconds if seconds > 0.0 else math.inf
    checksum = int(target[0].item()) + int(target[-1].item())
    del source, target
    torch.cuda.empty_cache()
    return {
        "buffer_gib": gib,
        "repeats": repeats,
        "warmup": warmup,
        "elapsed_ms": elapsed_ms,
        "traffic_gib": traffic_gib,
        "copy_bandwidth_gib_s": bandwidth_gib_s,
        "checksum": checksum,
    }


def _scenario_with_measured_bandwidth(
    scenario: dict[str, object],
    *,
    formula_bandwidth_gib_s: float,
    measured_bandwidth_gib_s: float,
    free_gib: float,
    safety_gib: float,
) -> dict[str, object]:
    synthetic_tok_s = float(scenario["synthetic_tokens_per_second"])
    measured_tok_s = synthetic_tok_s * measured_bandwidth_gib_s / formula_bandwidth_gib_s
    total_gib = float(scenario["total_gib"])
    return {
        **scenario,
        "measured_bandwidth_tok_s_proxy": measured_tok_s,
        "measured_bandwidth_speedup_vs_before": 0.0,
        "fits_current_free_memory_with_safety": total_gib <= max(0.0, free_gib - safety_gib),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    torch = _load_torch()
    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    preflight = {
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_runtime_available": True,
        "free_gib_before": free_bytes / BYTES_PER_GIB,
        "total_gib": total_bytes / BYTES_PER_GIB,
        "safety_gib": args.safety_gib,
    }

    sweep = run_t002_kv_margin_sweep(bandwidth_gib_per_second=args.formula_bandwidth_gib_s)
    bandwidth = _measure_cuda_copy_bandwidth(
        torch,
        gib=args.buffer_gib,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    free_after_bytes, _ = torch.cuda.mem_get_info()
    measured_bw = float(bandwidth["copy_bandwidth_gib_s"])
    scenarios = [
        _scenario_with_measured_bandwidth(
            scenario,
            formula_bandwidth_gib_s=args.formula_bandwidth_gib_s,
            measured_bandwidth_gib_s=measured_bw,
            free_gib=preflight["free_gib_before"],
            safety_gib=args.safety_gib,
        )
        for scenario in sweep["scenarios"]
    ]
    before_tok_s = float(scenarios[0]["measured_bandwidth_tok_s_proxy"])
    for scenario in scenarios:
        scenario["measured_bandwidth_speedup_vs_before"] = (
            float(scenario["measured_bandwidth_tok_s_proxy"]) / before_tok_s if before_tok_s > 0.0 else 0.0
        )

    strong = next(s for s in scenarios if s["name"] == "strong_weight_tighten_tiered_int4_kv_16k")
    tok_push = next(s for s in scenarios if s["name"] == "tok_s_push_exact_self_spec_int4_16k")
    overall = {
        "pass": bool(sweep["overall"]["pass"])
        and bool(strong["fits_12gib"])
        and float(tok_push["measured_bandwidth_speedup_vs_before"]) >= 1.20,
        "activation": "T002_REAL_CUDA_KV_BENCHMARK_ACTIVE",
        "measured_copy_bandwidth_gib_s": measured_bw,
        "before_measured_tok_s_proxy": before_tok_s,
        "strong_16k_measured_tok_s_proxy": float(strong["measured_bandwidth_tok_s_proxy"]),
        "tok_s_push_measured_tok_s_proxy": float(tok_push["measured_bandwidth_tok_s_proxy"]),
        "tok_s_push_speedup_vs_before": float(tok_push["measured_bandwidth_speedup_vs_before"]),
        "strong_16k_total_gib": float(strong["total_gib"]),
        "strong_16k_slack_gib": float(strong["slack_gib"]),
        "strong_16k_fits_current_free_memory_with_safety": bool(strong["fits_current_free_memory_with_safety"]),
    }
    return {
        "benchmark": "t002_kv_real_benchmark",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "activation": overall["activation"],
        "preflight": preflight,
        "bandwidth": bandwidth,
        "free_gib_after": free_after_bytes / BYTES_PER_GIB,
        "t002_sweep_overall": sweep["overall"],
        "candidate_summary": sweep["candidate_summary"],
        "kv_denominator": sweep["kv_denominator"],
        "scenarios": scenarios,
        "overall": overall,
        "limits": {
            "full_qwen3_weights_loaded": False,
            "real_model_quality_measured": False,
            "full_residency_allocation_attempted": False,
            "reason": "No local Qwen3-32B weights/runtime are available; this benchmark measures real CUDA bandwidth and KV allocation scale, then prices the T002 denominator.",
        },
    }


def write_report(result: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# T002 Real CUDA KV Benchmark",
        "",
        f"Timestamp: `{result['timestamp']}`",
        "",
        "## Activation",
        "",
        f"- `{result['activation']}`",
        "",
        "## Preflight",
        "",
    ]
    preflight = result["preflight"]
    lines.extend(
        [
            f"- device: `{preflight['device_name']}`",
            f"- torch: `{preflight['torch_version']}`",
            f"- free GiB before: `{preflight['free_gib_before']:.3f}`",
            f"- total GiB: `{preflight['total_gib']:.3f}`",
            f"- safety GiB: `{preflight['safety_gib']:.3f}`",
            "",
            "## Bandwidth",
            "",
        ]
    )
    bandwidth = result["bandwidth"]
    lines.extend(
        [
            f"- buffer GiB: `{bandwidth['buffer_gib']:.3f}`",
            f"- traffic GiB: `{bandwidth['traffic_gib']:.3f}`",
            f"- elapsed ms: `{bandwidth['elapsed_ms']:.3f}`",
            f"- copy bandwidth: `{bandwidth['copy_bandwidth_gib_s']:.3f} GiB/s`",
            "",
            "## Overall",
            "",
        ]
    )
    for key, value in result["overall"].items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.6f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Scenario Comparison",
            "",
            "| scenario | ctx | total GiB | slack GiB | measured tok/s proxy | speedup | fits current free+safety |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in result["scenarios"]:
        lines.append(
            "| {name} | {context_tokens} | {total_gib:.3f} | {slack_gib:.3f} | "
            "{measured_bandwidth_tok_s_proxy:.2f} | {measured_bandwidth_speedup_vs_before:.3f}x | "
            "{fits_current_free_memory_with_safety} |".format(**scenario)
        )
    lines.extend(
        [
            "",
            "This is a real CUDA bandwidth/allocation benchmark, not a real Qwen3-32B inference benchmark.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--buffer-gib", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--safety-gib", type=float, default=0.25)
    parser.add_argument("--formula-bandwidth-gib-s", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT / stamp)
    write_report(result, out_dir)
    print(result["activation"])
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "overall": result["overall"],
                "preflight": result["preflight"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
