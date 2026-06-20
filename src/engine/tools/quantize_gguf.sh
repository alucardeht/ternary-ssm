#!/bin/bash
# quantize_gguf.sh — Quantize a GGUF model with llama.cpp
# Usage: ./quantize_gguf.sh <input.gguf> <output.gguf> [QUANT_TYPE]
#
# QUANT_TYPE options:
#   Q2_K   — 2.5 bits per weight (closest to ternary ~1.6 bpw)
#   Q3_K_S — 3 bits, small variant
#   Q4_0   — 4.5 bits, balanced
#   F16    — No quantization (16-bit float baseline)
#
# Examples:
#   ./quantize_gguf.sh tiny-llama-6m-f16.gguf tiny-llama-6m-q2k.gguf Q2_K
#   ./quantize_gguf.sh model.gguf model-q4.gguf Q4_0

set -e

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../../../llama.cpp}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.gguf> <output.gguf> [QUANT_TYPE]"
    echo ""
    echo "QUANT_TYPE (default: Q2_K):"
    echo "  Q2_K   — 2.5 bpw (recommended, closest to ternary)"
    echo "  Q3_K_S — 3 bpw"
    echo "  Q4_0   — 4.5 bpw"
    echo "  F16    — No quantization (16-bit baseline)"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
QUANT_TYPE="${3:-Q2_K}"

if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT"
    exit 1
fi

QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize.exe"

if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "ERROR: llama-quantize not found at $QUANTIZE_BIN"
    echo ""
    echo "To build llama.cpp:"
    echo "  cd $LLAMA_CPP_DIR"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_AVX2=ON"
    echo "  cmake --build build --config Release -j"
    exit 1
fi

echo "Quantizing: $INPUT"
echo "  Method: $QUANT_TYPE"
echo "  Output: $OUTPUT"
echo ""

"$QUANTIZE_BIN" "$INPUT" "$OUTPUT" "$QUANT_TYPE"

INPUT_SIZE=$(stat -f%z "$INPUT" 2>/dev/null || stat -c%s "$INPUT" 2>/dev/null)
OUTPUT_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
INPUT_MB=$(echo "scale=2; $INPUT_SIZE / 1024 / 1024" | bc)
OUTPUT_MB=$(echo "scale=2; $OUTPUT_SIZE / 1024 / 1024" | bc)
RATIO=$(echo "scale=1; $OUTPUT_SIZE * 100 / $INPUT_SIZE" | bc)

echo ""
echo "✓ Quantization complete"
echo "  Input:  $INPUT_MB MB"
echo "  Output: $OUTPUT_MB MB"
echo "  Ratio:  $RATIO%"
echo ""
echo "Next: Run benchmark with llama-bench:"
echo "  $LLAMA_CPP_DIR/build/bin/llama-bench.exe -m $OUTPUT -t 1 -n 128 -r 5"
