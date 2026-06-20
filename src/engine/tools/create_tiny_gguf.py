#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path

import numpy as np


def check_gguf_package():
    try:
        import gguf
        return True
    except ImportError:
        return False


def create_tiny_gguf(output_path: str):
    try:
        from gguf import GGUFWriter
    except ImportError:
        print("ERROR: gguf package not found.")
        print("\nTo install, run:")
        print("  pip install gguf")
        print("\nOr if using conda:")
        print("  conda install -c conda-forge gguf")
        sys.exit(1)

    VOCAB = 1000
    D_MODEL = 256
    N_LAYERS = 4
    N_HEADS = 4
    D_FF = 683
    HEAD_DIM = D_MODEL // N_HEADS

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating tiny GGUF model...")
    print(f"  Architecture: Llama (Transformer)")
    print(f"  d_model: {D_MODEL}")
    print(f"  n_layers: {N_LAYERS}")
    print(f"  n_heads: {N_HEADS}")
    print(f"  vocab_size: {VOCAB}")
    print(f"  d_ff: {D_FF}")
    print()

    writer = GGUFWriter(str(output_path), "llama")

    writer.add_context_length(2048)
    writer.add_embedding_length(D_MODEL)
    writer.add_block_count(N_LAYERS)
    writer.add_feed_forward_length(D_FF)
    writer.add_head_count(N_HEADS)
    writer.add_head_count_kv(N_HEADS)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_vocab_size(VOCAB)

    print("Adding tokenizer metadata...")
    writer.add_tokenizer_model("llama")
    tokens = [f"tok_{i}".encode() for i in range(VOCAB)]
    scores = [0.0] * VOCAB
    token_types = [1] * VOCAB
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    print("Adding embedding layer...")
    writer.add_tensor(
        "token_embd.weight",
        np.random.randn(VOCAB, D_MODEL).astype(np.float16),
    )

    print(f"Adding {N_LAYERS} transformer blocks...")
    for i in range(N_LAYERS):
        prefix = f"blk.{i}"

        writer.add_tensor(
            f"{prefix}.attn_norm.weight",
            np.ones(D_MODEL, dtype=np.float32),
        )

        writer.add_tensor(
            f"{prefix}.attn_q.weight",
            np.random.randn(D_MODEL, D_MODEL).astype(np.float16),
        )

        writer.add_tensor(
            f"{prefix}.attn_k.weight",
            np.random.randn(D_MODEL, D_MODEL).astype(np.float16),
        )

        writer.add_tensor(
            f"{prefix}.attn_v.weight",
            np.random.randn(D_MODEL, D_MODEL).astype(np.float16),
        )

        writer.add_tensor(
            f"{prefix}.attn_output.weight",
            np.random.randn(D_MODEL, D_MODEL).astype(np.float16),
        )

        writer.add_tensor(
            f"{prefix}.ffn_norm.weight",
            np.ones(D_MODEL, dtype=np.float32),
        )

        writer.add_tensor(
            f"{prefix}.ffn_gate.weight",
            np.random.randn(D_FF, D_MODEL).astype(np.float16),
        )

        writer.add_tensor(
            f"{prefix}.ffn_up.weight",
            np.random.randn(D_FF, D_MODEL).astype(np.float16),
        )

        writer.add_tensor(
            f"{prefix}.ffn_down.weight",
            np.random.randn(D_MODEL, D_FF).astype(np.float16),
        )

        if (i + 1) % 2 == 0:
            print(f"  ... layer {i + 1}/{N_LAYERS}")

    print("Adding output layers...")
    writer.add_tensor(
        "output.weight",
        np.random.randn(VOCAB, D_MODEL).astype(np.float16),
    )

    writer.add_tensor(
        "output_norm.weight",
        np.ones(D_MODEL, dtype=np.float32),
    )

    print("Finalizing GGUF file...")
    writer.close()

    file_size_kb = os.path.getsize(output_path) / 1024
    file_size_mb = file_size_kb / 1024

    embedding_params = VOCAB * D_MODEL
    attn_params_per_layer = 5 * D_MODEL * D_MODEL
    ffn_params_per_layer = D_FF * D_MODEL + D_FF * D_MODEL + D_MODEL * D_FF
    norm_params_per_layer = 2 * D_MODEL
    params_per_layer = (
        attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    )
    total_transformer_params = N_LAYERS * params_per_layer
    output_params = VOCAB * D_MODEL + D_MODEL

    total_params = embedding_params + total_transformer_params + output_params
    total_params_m = total_params / 1e6

    print()
    print("=" * 60)
    print(f"✓ GGUF file created: {output_path}")
    print("=" * 60)
    print(f"File size:           {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")
    print(f"Total parameters:    {total_params:,} (~{total_params_m:.1f}M)")
    print()
    print("Parameter breakdown:")
    print(f"  Embedding:         {embedding_params:,}")
    print(f"  Transformer:       {total_transformer_params:,} ({N_LAYERS} × {params_per_layer:,})")
    print(f"    Per-layer:")
    print(f"      Attention:     {attn_params_per_layer:,}")
    print(f"      FFN:           {ffn_params_per_layer:,}")
    print(f"      Norms:         {norm_params_per_layer:,}")
    print(f"  Output head:       {output_params:,}")
    print()
    print("Next steps:")
    print()
    print("1. Quantize to Q2_K (closest to 2.5 bpw ternary):")
    print(f"   llama-quantize.exe {output_path} {str(output_path).replace('.gguf', '-q2k.gguf')} Q2_K")
    print()
    print("2. Test with llama-cli:")
    print(f"   llama-cli.exe -m {str(output_path).replace('.gguf', '-q2k.gguf')} -p 'hello' -n 10")
    print()
    print("3. Benchmark with single thread, 128 tokens:")
    print(f"   llama-bench.exe -m {str(output_path).replace('.gguf', '-q2k.gguf')} -t 1 -n 128 -r 5")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create a tiny 6M-parameter GGUF model for benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_tiny_gguf.py --output models/tiny-llama-6m-f16.gguf
  python create_tiny_gguf.py -o /tmp/model.gguf

The generated model can be quantized with llama.cpp:
  llama-quantize.exe model.gguf model-q2k.gguf Q2_K
""",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="tiny-llama-6m-f16.gguf",
        help="Output GGUF file path (default: tiny-llama-6m-f16.gguf)",
    )

    args = parser.parse_args()

    if not check_gguf_package():
        print("ERROR: gguf package not installed.")
        print("\nTo install, run:")
        print("  pip install gguf")
        sys.exit(1)

    create_tiny_gguf(args.output)


if __name__ == "__main__":
    main()
