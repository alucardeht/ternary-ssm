"""CPU baseline benchmark for model_v5. Run: python -m src.model_v5.benchmarks.bench_cpu_baseline"""

import sys
import os
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_v5.config import ModelConfig
from src.model_v5.model import ModelV2

WARMUP = 50
ITERS = 500

def bench_timing(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        t_start = time.perf_counter_ns()
        fn()
        t_end = time.perf_counter_ns()
        times.append((t_end - t_start) / 1000.0)

    avg_us = sum(times) / len(times)
    min_us = min(times)
    max_us = max(times)
    tok_s = 1e6 / avg_us if avg_us > 0 else 0

    return avg_us, min_us, max_us, tok_s

def main():
    torch.set_num_threads(1)

    config = ModelConfig(
        d_model=256,
        n_layer=4,
        vocab_size=1000,
        d_state=16,
        expand=2,
        dt_rank=16,
        n_heads=4,
        ssd_n_heads=8,
        window_size=128,
        swa_layer_indices=[3],
        use_ssd=True,
        seq_len=128,
        use_ternary=True,
    )

    print("=== CPU Baseline Benchmark (PyTorch) ===\n")
    print(f"Config: d_model={config.d_model}, n_layer={config.n_layer}, vocab={config.vocab_size}")
    print(f"        d_state={config.d_state}, expand={config.expand}, n_heads={config.n_heads}")
    print(f"        ssd_n_heads={config.ssd_n_heads}, window={config.window_size}\n")

    model = ModelV2(config)
    model.eval()
    torch.no_grad().__enter__()

    print(f"  {'Component':<30} {'Avg (us)':>12} {'Min (us)':>12} {'Max (us)':>12} {'Est tok/s':>12}")
    print(f"  {'-'*80}")

    x_vec = torch.randn(1, 1, config.d_model, dtype=torch.float32)

    def bench_rmsnorm():
        for layer in model.layers:
            norm = layer.norm
            norm(x_vec)

    avg_us, min_us, max_us, tok_s = bench_timing(bench_rmsnorm)
    print(f"  {'RMSNorm':<30} {avg_us:>12.2f} {min_us:>12.2f} {max_us:>12.2f} {tok_s:>12.0f}")

    ssm_layer = None
    swa_layer = None
    for layer in model.layers:
        if not layer.is_swa and ssm_layer is None:
            ssm_layer = layer
        if layer.is_swa:
            swa_layer = layer

    if ssm_layer is not None:
        def bench_ssm():
            with torch.no_grad():
                ssm_layer.block(x_vec)

        avg_us, min_us, max_us, tok_s = bench_timing(bench_ssm)
        print(f"  {'SSM Layer':<30} {avg_us:>12.2f} {min_us:>12.2f} {max_us:>12.2f} {tok_s:>12.0f}")

    if swa_layer is not None:
        def bench_swa():
            with torch.no_grad():
                swa_layer.block(x_vec)

        avg_us, min_us, max_us, tok_s = bench_timing(bench_swa)
        print(f"  {'SWA Layer':<30} {avg_us:>12.2f} {min_us:>12.2f} {max_us:>12.2f} {tok_s:>12.0f}")

    def bench_lm_head():
        hidden = torch.randn(1, 1, config.d_model, dtype=torch.float32)
        with torch.no_grad():
            if model.lm_head is None:
                logits = torch.nn.functional.linear(hidden, model.embedding.weight)
            else:
                logits = model.lm_head(hidden)

    avg_us, min_us, max_us, tok_s = bench_timing(bench_lm_head)
    print(f"  {'LM Head':<30} {avg_us:>12.2f} {min_us:>12.2f} {max_us:>12.2f} {tok_s:>12.0f}")

    def bench_forward():
        token = torch.tensor([[42]], dtype=torch.long)
        with torch.no_grad():
            model(token)

    avg_us, min_us, max_us, tok_s = bench_timing(bench_forward)
    print(f"  {'Full Forward':<30} {avg_us:>12.2f} {min_us:>12.2f} {max_us:>12.2f} {tok_s:>12.0f}")

    print("\n--- Summary ---")
    print(f"Total parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Device: CPU (single-threaded)")

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        print(f"Peak RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
    except ImportError:
        pass

if __name__ == '__main__':
    main()
