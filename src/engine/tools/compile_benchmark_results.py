#!/usr/bin/env python3

import re
import argparse
from pathlib import Path
from datetime import datetime


def parse_our_engine_results(text):
    results = {}

    # Parse GEMV results
    gemv_match = re.search(
        r"Ternary GEMV\s+(\d+)x(\d+):\s+([\d.]+)\s+us",
        text,
    )
    if gemv_match:
        size_a, size_b, latency = gemv_match.groups()
        results["gemv_256x512"] = {
            "latency_us": float(latency),
            "ops": f"{size_a}×{size_b}",
        }

    e2e_match = re.search(
        r"E2E forward.*?(\d+\.\d+)\s+us",
        text,
        re.DOTALL,
    )
    if e2e_match:
        results["e2e_latency_us"] = float(e2e_match.group(1))

    tok_s_match = re.search(r"(\d+)\s+tok/s", text)
    if tok_s_match:
        results["tokens_per_sec"] = int(tok_s_match.group(1))

    vram_match = re.search(r"Peak VRAM:\s+([\d.]+)\s+MB", text)
    if vram_match:
        results["peak_vram_mb"] = float(vram_match.group(1))

    return results


def parse_llama_cpp_bench(text):
    results = {}

    lines = text.split("\n")
    for line in lines:
        if "model_size" in line or "tokens/sec" in line or "time_per_token" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if "tokens/sec" in part and i > 0:
                    try:
                        results["tokens_per_sec"] = float(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
                if "ms" in part and "time_per_token" in line:
                    try:
                        results["latency_ms"] = float(part.replace("ms", ""))
                    except ValueError:
                        pass

    tok_s_match = re.search(r"(\d+\.?\d*)\s+tokens/sec", text)
    if tok_s_match:
        results["tokens_per_sec"] = float(tok_s_match.group(1))

    return results


def format_benchmark_report(our_results, llama_q2k_results, llama_f16_results):
    report = []
    report.append("=" * 80)
    report.append("BENCHMARK COMPARISON: Our Ternary Engine vs llama.cpp")
    report.append("=" * 80)
    report.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("Models:")
    report.append("  Our Engine:  6M ternary (d=256, 4 layers, SSM+Attention)")
    report.append("  llama.cpp:   6M Transformer (d=256, 4 layers, Llama architecture)")
    report.append("")

    report.append("Hardware:")
    report.append("  CPU: Intel Core Ultra 5 225F (6 P-cores, no HT)")
    report.append("  RAM: 32GB DDR5")
    report.append("  SIMD: AVX2, FMA, POPCNT")
    report.append("")

    report.append("-" * 80)
    report.append("KERNEL-LEVEL COMPARISON (GEMV latency)")
    report.append("-" * 80)
    report.append("")

    if "gemv_256x512" in our_results:
        our_gemv = our_results["gemv_256x512"]["latency_us"]
        report.append(f"Our Ternary GEMV (256×512):  {our_gemv:.2f} us")
        report.append("  (Uses add/subtract operations on packed ternary weights)")
    else:
        report.append("Our Ternary GEMV:  (no data)")

    report.append("")
    report.append(
        "llama.cpp Benchmark (run 'test-backend-ops.exe perf' for detailed results)"
    )
    report.append("")

    report.append("-" * 80)
    report.append("END-TO-END INFERENCE (128 token generation)")
    report.append("-" * 80)
    report.append("")

    if "tokens_per_sec" in our_results:
        our_tok_s = our_results["tokens_per_sec"]
        report.append(f"Our Engine:           {our_tok_s} tok/s")
    else:
        report.append("Our Engine:           (no data)")

    if "tokens_per_sec" in llama_q2k_results:
        llama_q2k_tok_s = llama_q2k_results["tokens_per_sec"]
        report.append(f"llama.cpp (Q2_K):     {llama_q2k_tok_s:.1f} tok/s")
    else:
        report.append("llama.cpp (Q2_K):     (no data)")

    if "tokens_per_sec" in llama_f16_results:
        llama_f16_tok_s = llama_f16_results["tokens_per_sec"]
        report.append(f"llama.cpp (F16):      {llama_f16_tok_s:.1f} tok/s")
    else:
        report.append("llama.cpp (F16):      (no data)")

    report.append("")

    if "tokens_per_sec" in our_results and "tokens_per_sec" in llama_q2k_results:
        our_tok_s = our_results["tokens_per_sec"]
        llama_q2k_tok_s = llama_q2k_results["tokens_per_sec"]
        speedup = our_tok_s / llama_q2k_tok_s
        report.append(f"Speedup (ours vs Q2_K):  {speedup:.1f}x")

    report.append("")

    report.append("-" * 80)
    report.append("MEMORY USAGE")
    report.append("-" * 80)
    report.append("")

    if "peak_vram_mb" in our_results:
        report.append(f"Our Engine Peak VRAM:  {our_results['peak_vram_mb']:.1f} MB")
    else:
        report.append("Our Engine Peak VRAM:  (no data)")

    report.append("")

    report.append("-" * 80)
    report.append("ANALYSIS")
    report.append("-" * 80)
    report.append("")

    report.append("What this benchmark measures:")
    report.append("  - Kernel efficiency: Raw matrix-vector multiply throughput")
    report.append("  - E2E efficiency: Framework overhead + kernel speed")
    report.append("  - Memory footprint: RAM required for model + activations")
    report.append("")

    report.append("Fair comparisons:")
    report.append("  - Both models have ~6M parameters and ~256 dimensions")
    report.append("  - Both run single-threaded (CPU only)")
    report.append("  - Token generation only (not prompt processing)")
    report.append("")

    report.append("Unfair comparisons:")
    report.append("  - Architecture differs (SSM+Attention vs pure Transformer)")
    report.append("  - Different quantization (packed ternary vs Q2_K)")
    report.append("  - Our engine lacks prompt batching (single-token generation)")
    report.append("  - llama.cpp has years of optimization for large dimensions")
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Compile benchmark results into formatted report"
    )
    parser.add_argument(
        "raw_results",
        type=str,
        help="Path to raw benchmark results file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    results_file = Path(args.raw_results)
    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        return 1

    raw_text = results_file.read_text()

    our_results = parse_our_engine_results(raw_text)
    llama_q2k_results = parse_llama_cpp_bench(
        raw_text[raw_text.find("llama.cpp bench (Q2_K)") :]
        if "llama.cpp bench (Q2_K)" in raw_text
        else ""
    )
    llama_f16_results = parse_llama_cpp_bench(
        raw_text[raw_text.find("llama.cpp bench (F16)") :]
        if "llama.cpp bench (F16)" in raw_text
        else ""
    )

    report = format_benchmark_report(our_results, llama_q2k_results, llama_f16_results)

    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        print(f"Report written to: {output_file}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
