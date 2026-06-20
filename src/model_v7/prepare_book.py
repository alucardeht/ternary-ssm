"""CLI entry point for Source->Signal book data pipeline."""

import argparse
import sys
from pathlib import Path
from .data_pipeline import run_pipeline
from .data_pipeline.config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description='Prepare book data with Source->Signal pipeline'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input text file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: data/books/<input_filename>/)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='data/wikitext103/tokenizer.json',
        help='Tokenizer path (default: data/wikitext103/tokenizer.json)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Optional model checkpoint for perplexity scoring'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=512,
        help='Sequence length (default: 512)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=0,
        help='Max examples (default: 0 = no limit)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        output_dir = Path('data') / 'books' / input_path.stem
    else:
        output_dir = Path(args.output)

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found: {args.tokenizer}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        seq_len=args.seq_len,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    print(f"Source->Signal Book Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    if args.model:
        print(f"Model (perplexity scoring): {args.model}")
    print()

    result = run_pipeline(
        input_path=str(input_path),
        output_dir=str(output_dir),
        model_path=args.model,
        tokenizer_path=str(tokenizer_path),
        config=config
    )

    print("\nPipeline Complete")
    print(f"Chunks: {result.n_chunks}")
    print(f"Scored chunks: {result.n_scored_chunks}")
    print(f"Facts: {result.n_facts}")
    print(f"Generated examples: {result.n_examples}")
    print(f"Filtered examples: {result.n_filtered}")
    print(f"Output: {result.output_path}")
    print()
    print("Timing (seconds):")
    for stage, duration in result.timing.items():
        print(f"  {stage}: {duration:.2f}s")


if __name__ == '__main__':
    main()
