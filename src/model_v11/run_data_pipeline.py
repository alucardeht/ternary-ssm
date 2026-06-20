"""CLI wrapper for the Source->Signal data pipeline (Phase 14)."""

import argparse
import time
from pathlib import Path

from src.model_v11.data_pipeline import run_pipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description='Run Source->Signal data pipeline'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input .txt file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--tokenizer',
        default=None,
        help='Path to tokenizer (for token encoding)'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Path to HF model (for perplexity scoring)'
    )
    parser.add_argument(
        '--min-chunk-size',
        type=int,
        default=100,
        help='Min chunk size in chars (default: 100)'
    )
    parser.add_argument(
        '--max-chunk-size',
        type=int,
        default=2000,
        help='Max chunk size in chars (default: 2000)'
    )
    parser.add_argument(
        '--top-fraction',
        type=float,
        default=0.25,
        help='Top fraction of chunks to keep (default: 0.25)'
    )
    args = parser.parse_args()

    config = PipelineConfig()
    if args.min_chunk_size != 100:
        config.min_chunk_size = args.min_chunk_size
    if args.max_chunk_size != 2000:
        config.max_chunk_size = args.max_chunk_size
    if args.top_fraction != 0.25:
        config.top_fraction = args.top_fraction

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Tokenizer: {args.tokenizer or 'None (no token encoding)'}")
    print(f"Model: {args.model or 'None (heuristic scoring only)'}")
    print()

    t0 = time.time()
    result = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        config=config,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Chunks parsed: {result.n_chunks}")
    print(f"Chunks scored (kept): {result.n_scored_chunks}")
    print(f"Facts extracted: {result.n_facts}")
    print(f"Examples generated: {result.n_examples}")
    print(f"Examples after filter: {result.n_filtered}")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")

    if hasattr(result, 'timing') and result.timing:
        print("\nTiming breakdown:")
        for stage, duration in result.timing.items():
            print(f"  {stage}: {duration:.2f}s")


if __name__ == '__main__':
    main()
