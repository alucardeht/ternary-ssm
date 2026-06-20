"""Prepare combined WikiText + book dataset for training.

Two-mode tool for integrating book data into training pipeline:
1. tokenize-books: Tokenize book_dirs, split train/test per book, concatenate
2. combine: Merge wiki dataset with book dataset at target ratio

Output: data/combined/{train.bin, test.bin, val.bin, ref_losses.bin}

Usage:
  python -m model_v7.prepare_combined_dataset tokenize-books \
    --book-dirs data/books/advanced_algorithms data/books/algorithms_notes \
    --tokenizer data/wikitext103/tokenizer.json --output-dir data/books_tokenized

  python -m model_v7.prepare_combined_dataset combine \
    --wiki-dir data/wikitext103 --books-dir data/books_tokenized \
    --output-dir data/combined --book-ratio 0.10
"""
import argparse
import math
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def tokenize_books(book_dirs: list[str], tokenizer_path: str, output_dir: str,
                   test_fraction: float = 0.10):
    """Tokenize books, split train/test per book, save concatenated tokens.

    Args:
        book_dirs: List of directory paths, each containing content.txt
        tokenizer_path: Path to tokenizer.json
        output_dir: Where to save train.bin and test.bin
        test_fraction: Fraction of tokens per book reserved for test set
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    batch_size = 1000

    all_train_tokens = []
    all_test_tokens = []
    per_book_stats = []

    print("=" * 70)
    print("Tokenizing Books")
    print("=" * 70)

    for book_dir in book_dirs:
        book_path = Path(book_dir)
        content_file = book_path / 'content.txt'

        if not content_file.exists():
            print(f"  ⚠ {content_file} not found, skipping")
            continue

        print(f"\n  --- {book_path.name} ---")

        with open(content_file, 'r', encoding='utf-8') as f:
            text = f.read()

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        book_tokens = []
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i:i + batch_size]
            encoded = tokenizer.encode_batch(batch)

            for enc in encoded:
                book_tokens.extend(enc.ids)

            if (i // batch_size) % 5 == 0:
                print(f"    Tokenized {i + len(batch):,}/{len(paragraphs):,} paragraphs "
                      f"({len(book_tokens):,} tokens)")

        split_idx = int(len(book_tokens) * (1 - test_fraction))
        train_portion = book_tokens[:split_idx]
        test_portion = book_tokens[split_idx:]

        all_train_tokens.extend(train_portion)
        all_test_tokens.extend(test_portion)

        per_book_stats.append({
            'name': book_path.name,
            'total': len(book_tokens),
            'train': len(train_portion),
            'test': len(test_portion),
        })

        print(f"    Total: {len(book_tokens):,} tokens")
        print(f"    Train: {len(train_portion):,} | Test: {len(test_portion):,}")

    train_array = np.array(all_train_tokens, dtype=np.uint16)
    test_array = np.array(all_test_tokens, dtype=np.uint16)

    train_path = output_dir / 'train.bin'
    test_path = output_dir / 'test.bin'

    train_array.tofile(train_path)
    test_array.tofile(test_path)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for stats in per_book_stats:
        print(f"  {stats['name']:30s} train: {stats['train']:10,}  test: {stats['test']:10,}")

    print(f"\n  Total train: {len(all_train_tokens):,} tokens -> {train_path}")
    print(f"  Total test:  {len(all_test_tokens):,} tokens -> {test_path}")


def combine_datasets(wiki_dir: str, books_dir: str, output_dir: str,
                     book_target_ratio: float = 0.10):
    """Combine wiki and book datasets at target ratio.

    Oversamples book data to achieve target ratio:
      target_ratio = book_tokens / (wiki_tokens + book_tokens)

    Args:
        wiki_dir: Directory with train.bin, test.bin, val.bin, ref_losses.bin
        books_dir: Directory with train.bin, test.bin from tokenize_books
        output_dir: Where to save combined dataset
        book_target_ratio: Target fraction of final dataset from books
    """
    wiki_dir = Path(wiki_dir)
    books_dir = Path(books_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Combining Datasets")
    print("=" * 70)

    wiki_train_path = wiki_dir / 'train.bin'
    books_train_path = books_dir / 'train.bin'

    if not wiki_train_path.exists():
        raise FileNotFoundError(f"{wiki_train_path} not found")
    if not books_train_path.exists():
        raise FileNotFoundError(f"{books_train_path} not found")

    wiki_data = np.memmap(wiki_train_path, dtype=np.uint16, mode='r')
    books_data = np.memmap(books_train_path, dtype=np.uint16, mode='r')

    wiki_tokens = len(wiki_data)
    book_tokens = len(books_data)

    print(f"\n  Wiki tokens:  {wiki_tokens:,}")
    print(f"  Book tokens:  {book_tokens:,}")
    print(f"  Target book ratio: {book_target_ratio:.1%}")

    oversample_factor = math.ceil(
        wiki_tokens * book_target_ratio /
        (book_tokens * (1 - book_target_ratio))
    )

    print(f"  Oversample factor: {oversample_factor}x")

    combined_train_path = output_dir / 'train.bin'
    with open(combined_train_path, 'wb') as f:
        wiki_data.tofile(f)
        for _ in range(oversample_factor):
            books_data.tofile(f)

    combined_train_data = np.memmap(combined_train_path, dtype=np.uint16, mode='r')
    actual_ratio = book_tokens * oversample_factor / len(combined_train_data)

    wiki_ref_path = wiki_dir / 'ref_losses.bin'
    books_ref_path = books_dir / 'ref_losses.bin'

    if not wiki_ref_path.exists():
        raise FileNotFoundError(
            f"{wiki_ref_path} not found. WikiText dataset needs ref_losses.bin"
        )

    if not books_ref_path.exists():
        raise FileNotFoundError(
            f"{books_ref_path} not found. Book dataset needs ref_losses.bin. "
            "Run tokenize-books first or compute reference losses."
        )

    wiki_ref = np.memmap(wiki_ref_path, dtype=np.float16, mode='r')
    books_ref = np.memmap(books_ref_path, dtype=np.float16, mode='r')

    combined_ref_path = output_dir / 'ref_losses.bin'
    with open(combined_ref_path, 'wb') as f:
        wiki_ref.tofile(f)
        for _ in range(oversample_factor):
            books_ref.tofile(f)

    shutil.copy2(wiki_dir / 'val.bin', output_dir / 'val.bin')
    shutil.copy2(wiki_dir / 'test.bin', output_dir / 'test.bin')

    print(f"\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Combined train tokens: {len(combined_train_data):,}")
    print(f"  Actual book ratio: {actual_ratio:.1%}")
    print(f"\n  Saved:")
    print(f"    {combined_train_path} ({len(combined_train_data):,} tokens)")
    print(f"    {combined_ref_path}")
    print(f"    {output_dir / 'val.bin'} (copied)")
    print(f"    {output_dir / 'test.bin'} (copied)")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare combined WikiText + book dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    tokenize_parser = subparsers.add_parser(
        'tokenize-books',
        help='Tokenize book files and split train/test',
    )
    tokenize_parser.add_argument(
        '--book-dirs',
        nargs='+',
        required=True,
        help='Directory paths containing content.txt files',
    )
    tokenize_parser.add_argument(
        '--tokenizer',
        required=True,
        help='Path to tokenizer.json',
    )
    tokenize_parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for train.bin and test.bin',
    )
    tokenize_parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.10,
        help='Fraction of tokens per book for test set (default: 0.10)',
    )

    combine_parser = subparsers.add_parser(
        'combine',
        help='Combine wiki and book datasets at target ratio',
    )
    combine_parser.add_argument(
        '--wiki-dir',
        required=True,
        help='WikiText dataset directory',
    )
    combine_parser.add_argument(
        '--books-dir',
        required=True,
        help='Book dataset directory (from tokenize-books)',
    )
    combine_parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for combined dataset',
    )
    combine_parser.add_argument(
        '--book-ratio',
        type=float,
        default=0.10,
        help='Target fraction of dataset from books (default: 0.10)',
    )

    args = parser.parse_args()

    if args.command == 'tokenize-books':
        tokenize_books(
            args.book_dirs,
            args.tokenizer,
            args.output_dir,
            args.test_fraction,
        )
    elif args.command == 'combine':
        combine_datasets(
            args.wiki_dir,
            args.books_dir,
            args.output_dir,
            args.book_ratio,
        )


if __name__ == '__main__':
    main()
