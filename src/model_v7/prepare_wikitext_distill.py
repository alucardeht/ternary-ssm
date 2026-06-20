"""Prepare WikiText-103 for knowledge distillation experiments.

One-time script that:
1. Downloads WikiText-103 via HuggingFace datasets
2. Loads TinyLlama's tokenizer (pre-trained from HuggingFace)
3. Tokenizes train/val/test, concatenates, saves as .bin (numpy memmap-compatible)

Output: data/wikitext103_distill/{train.bin, val.bin, test.bin}

Usage: python -m model_v7.prepare_wikitext_distill [--output_dir data/wikitext103_distill]
"""
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_and_save(tokenizer, texts: list[str], output_path: Path):
    """Tokenize texts in batches, write tokens to .bin file incrementally."""
    batch_size = 1000
    total_tokens = 0

    with open(output_path, 'wb') as f:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t for t in batch if t.strip()]
            if not batch:
                continue

            batch_tokens = []
            for text in batch:
                ids = tokenizer.encode(text, add_special_tokens=False)
                batch_tokens.extend(ids)

            chunk = np.array(batch_tokens, dtype=np.uint16)
            chunk.tofile(f)
            total_tokens += len(batch_tokens)

            if (i // batch_size) % 10 == 0:
                print(f"  Tokenized {i + len(batch):,}/{len(texts):,} texts ({total_tokens:,} tokens)")

    print(f"  Saved {total_tokens:,} tokens to {output_path}")
    return total_tokens


def main(output_dir: str):
    data_dir = Path(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download WikiText-103
    print("=" * 60)
    print("Step 1: Downloading WikiText-103")
    print("=" * 60)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    print(f"  Train: {len(dataset['train']):,} rows")
    print(f"  Val:   {len(dataset['validation']):,} rows")
    print(f"  Test:  {len(dataset['test']):,} rows")

    # 2. Load TinyLlama tokenizer
    print("\n" + "=" * 60)
    print("Step 2: Loading TinyLlama tokenizer")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"  Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # 3. Tokenize all splits
    print("\n" + "=" * 60)
    print("Step 3: Tokenizing and saving splits")
    print("=" * 60)

    splits = {
        'train': [row['text'] for row in dataset['train']],
        'val': [row['text'] for row in dataset['validation']],
        'test': [row['text'] for row in dataset['test']],
    }

    stats = {}
    file_sizes = {}
    for split_name, texts in splits.items():
        print(f"\n  --- {split_name} ---")
        output_path = data_dir / f'{split_name}.bin'
        n_tokens = tokenize_and_save(tokenizer, texts, output_path)
        stats[split_name] = n_tokens
        file_sizes[split_name] = output_path.stat().st_size

    # Summary
    print("\n" + "=" * 60)
    print("Done! Summary:")
    print("=" * 60)
    for split_name, n_tokens in stats.items():
        file_size_mb = file_sizes[split_name] / (1024 ** 2)
        print(f"  {split_name}: {n_tokens:,} tokens ({file_size_mb:.2f} MB)")
    print(f"  Tokenizer: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (vocab_size={tokenizer.vocab_size})")
    print(f"  Output directory: {data_dir.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare WikiText-103 for knowledge distillation using TinyLlama tokenizer"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/wikitext103_distill',
        help='Output directory for tokenized data (default: data/wikitext103_distill)'
    )
    args = parser.parse_args()

    main(args.output_dir)
