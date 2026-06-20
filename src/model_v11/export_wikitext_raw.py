"""Export WikiText-103 train split as raw text for pipeline processing."""

import argparse
from pathlib import Path
from datasets import load_dataset


def export_raw_text(output_path: str, split: str = 'train'):
    """Export WikiText-103 raw text to a .txt file."""
    print(f"Loading WikiText-103 {split} split...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    n_chars = 0
    with open(output, 'w', encoding='utf-8') as f:
        for item in ds:
            text = item['text']
            if text.strip():
                f.write(text + '\n')
                n_lines += 1
                n_chars += len(text)

    file_size_mb = output.stat().st_size / 1024 / 1024
    print(f"Exported {n_lines:,} lines, {n_chars:,} chars to {output}")
    print(f"File size: {file_size_mb:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export WikiText-103 as raw text'
    )
    parser.add_argument(
        '--output',
        default='data/wikitext103/train_raw.txt',
        help='Output file path (default: data/wikitext103/train_raw.txt)'
    )
    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split (default: train)'
    )
    args = parser.parse_args()
    export_raw_text(args.output, args.split)
