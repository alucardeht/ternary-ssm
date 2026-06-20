#!/usr/bin/env python3
"""
Convert HuggingFace tokenizer.json to .tvoc binary format.

Usage: python convert_tokenizer.py <input.json> <output.tvoc>
"""

import json
import struct
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple


TVOC_MAGIC = b'TVOC'
TVOC_VERSION = 1
TVOC_HEADER_SIZE = 64
TVOC_FLAG_BYTE_LEVEL = 1 << 0


def load_tokenizer_json(path: str) -> Tuple[Dict[str, int], List[str], List[Tuple[int, int, int]], Dict[str, int]]:
    """Load tokenizer.json and extract vocab, merges, special tokens."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = data.get('model', {})
    vocab = model.get('vocab', {})
    merges_list = model.get('merges', [])

    special_tokens = {}
    for entry in data.get('added_tokens', []):
        token_str = entry.get('content', '')
        token_id = entry.get('id', -1)
        if token_id >= 0:
            special_tokens[token_str] = token_id

    flags = 0
    pre_tokenizer = data.get('pre_tokenizer', {})
    if isinstance(pre_tokenizer, dict):
        if pre_tokenizer.get('type') == 'ByteLevel':
            flags |= TVOC_FLAG_BYTE_LEVEL

    id_to_str = {v: k for k, v in vocab.items()}

    processed_merges = []
    for merge_str in merges_list:
        parts = merge_str.split()
        if len(parts) != 2:
            continue
        str_a, str_b = parts
        combined = str_a + str_b
        if combined in vocab:
            result_id = vocab[combined]
            if str_a in vocab and str_b in vocab:
                token_a = vocab[str_a]
                token_b = vocab[str_b]
                processed_merges.append((token_a, token_b, result_id))

    return vocab, id_to_str, processed_merges, special_tokens, flags


def get_special_token_id(special_tokens: Dict[str, int], default: int, keys: List[str]) -> int:
    """Get special token ID by trying multiple key names."""
    for key in keys:
        if key in special_tokens:
            return special_tokens[key]
    return default


def build_vocab_data(vocab: Dict[str, int]) -> Tuple[bytes, List[int]]:
    """Build concatenated vocab data and offset table."""
    id_to_str = {v: k for k, v in vocab.items()}

    offsets = []
    data_parts = []
    current_offset = 0

    for token_id in range(len(vocab)):
        if token_id not in id_to_str:
            offsets.append(current_offset)
            continue

        token_str = id_to_str[token_id]
        token_bytes = token_str.encode('utf-8')
        offsets.append(current_offset)
        data_parts.append(token_bytes)
        current_offset += len(token_bytes)

    offsets.append(current_offset)

    vocab_data = b''.join(data_parts)
    return vocab_data, offsets[:-1]


def write_tvoc(output_path: str, vocab: Dict[str, int], id_to_str: Dict[int, str],
               merges: List[Tuple[int, int, int]], special_tokens: Dict[str, int], flags: int) -> None:
    """Write .tvoc binary format."""

    vocab_size = len(vocab)
    n_merges = len(merges)

    pad_id = get_special_token_id(special_tokens, 0, ['<|pad|>', '[PAD]'])
    unk_id = get_special_token_id(special_tokens, 1, ['<|unk|>', '[UNK]'])
    bos_id = get_special_token_id(special_tokens, 2, ['<|startoftext|>', '<|begin_of_text|>', '[BOS]'])
    eos_id = get_special_token_id(special_tokens, 3, ['<|endoftext|>', '<|end_of_text|>', '[EOS]'])

    vocab_data, vocab_offsets = build_vocab_data(vocab)
    vocab_data_size = len(vocab_data)

    with open(output_path, 'wb') as f:
        header = struct.pack('<4sIIIIIIII24s',
            TVOC_MAGIC,
            TVOC_VERSION,
            vocab_size,
            n_merges,
            pad_id,
            unk_id,
            bos_id,
            eos_id,
            flags,
            vocab_data_size,
            b'\x00' * 24
        )
        f.write(header)

        for offset in vocab_offsets:
            f.write(struct.pack('<I', offset))

        f.write(vocab_data)

        for token_a, token_b, result_id in merges:
            f.write(struct.pack('<III', token_a, token_b, result_id))

    print(f"Wrote {output_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Merges: {n_merges}")
    print(f"  Special tokens: pad={pad_id}, unk={unk_id}, bos={bos_id}, eos={eos_id}")
    print(f"  Vocab data: {vocab_data_size} bytes")
    print(f"  Byte-level BPE: {'yes' if (flags & TVOC_FLAG_BYTE_LEVEL) else 'no'}")


def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace tokenizer.json to .tvoc format')
    parser.add_argument('input', help='Input tokenizer.json path')
    parser.add_argument('output', help='Output .tvoc path')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    try:
        vocab, id_to_str, merges, special_tokens, flags = load_tokenizer_json(args.input)
        write_tvoc(args.output, vocab, id_to_str, merges, special_tokens, flags)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
