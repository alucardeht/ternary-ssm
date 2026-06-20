#!/usr/bin/env python3
"""Pre-tokenize and pack datasets into binary mmap format.

Usage:
    python -m src.model_v7.prepare_binary_dataset \
        --input data/wikitext103/train.txt \
        --output data/wikitext103/train_packed.bin \
        --tokenizer data/tokenizer.json \
        --seq-len 256 \
        --vocab-size 32000

Output:
    - {output}.bin: uint16 mmap array of packed token IDs
    - {output}.meta: JSON metadata (n_tokens, seq_len, tokenizer_path)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer


def prepare_binary_dataset(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    seq_len: int,
    vocab_size: int,
):
    """Tokenize and save dataset as binary mmap.

    Args:
        input_path: Path to text file (raw text or .txt)
        output_path: Path to output .bin file
        tokenizer_path: Path to tokenizer.json
        seq_len: Sequence length (used for metadata, not packing here)
        vocab_size: Vocabulary size (validation only)
    """
    print(f"[PREP] Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    if tokenizer.get_vocab_size() != vocab_size:
        print(f"WARNING: Tokenizer vocab_size={tokenizer.get_vocab_size()} != config vocab_size={vocab_size}")

    print(f"[PREP] Reading input from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"[PREP] Tokenizing {len(text)} characters...")
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids

    print(f"[PREP] Tokenized to {len(token_ids)} tokens")

    # Convert to uint16 (vocab < 65536)
    if vocab_size > 65535:
        dtype = np.uint32
    else:
        dtype = np.uint16

    token_array = np.array(token_ids, dtype=dtype)

    print(f"[PREP] Saving binary to {output_path}")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    token_array.tofile(output_file)

    # Save metadata
    meta = {
        'n_tokens': len(token_ids),
        'seq_len': seq_len,
        'vocab_size': vocab_size,
        'tokenizer_path': str(tokenizer_path),
        'dtype': str(dtype),
    }
    meta_path = output_file.with_suffix('.meta')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[PREP] Done. Metadata saved to {meta_path}")
    print(f"[PREP] Total size: {token_array.nbytes / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize dataset to binary mmap format')
    parser.add_argument('--input', required=True, help='Input text file')
    parser.add_argument('--output', required=True, help='Output .bin file')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer .json file')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size')

    args = parser.parse_args()

    prepare_binary_dataset(
        input_path=args.input,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )


if __name__ == '__main__':
    main()
