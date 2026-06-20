"""Prepare WikiText-103 for training.

One-time script that:
1. Downloads WikiText-103 via HuggingFace datasets
2. Trains a BPE tokenizer (vocab_size=32000) on the train split
3. Tokenizes train/val/test, concatenates, saves as .bin (numpy memmap-compatible)

Output: data/wikitext103/{tokenizer.json, train.bin, val.bin, test.bin}

Usage: python -m model_v5.prepare_wikitext
"""
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'wikitext103'
VOCAB_SIZE = 32000


def train_tokenizer(texts: list[str]) -> Tokenizer:
    """Train a BPE tokenizer on the given texts."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processor for BOS/EOS
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    return tokenizer


def tokenize_and_save(tokenizer: Tokenizer, texts: list[str], output_path: Path):
    """Tokenize texts in batches, write tokens to .bin file incrementally."""
    batch_size = 1000
    total_tokens = 0

    with open(output_path, 'wb') as f:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t for t in batch if t.strip()]
            if not batch:
                continue
            encoded = tokenizer.encode_batch(batch)

            batch_tokens = []
            for enc in encoded:
                batch_tokens.extend(enc.ids)

            chunk = np.array(batch_tokens, dtype=np.uint16)
            chunk.tofile(f)
            total_tokens += len(batch_tokens)

            if (i // batch_size) % 10 == 0:
                print(f"  Tokenized {i + len(batch):,}/{len(texts):,} texts ({total_tokens:,} tokens)")

    print(f"  Saved {total_tokens:,} tokens to {output_path}")
    return total_tokens


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download WikiText-103
    print("=" * 60)
    print("Step 1: Downloading WikiText-103")
    print("=" * 60)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    print(f"  Train: {len(dataset['train']):,} rows")
    print(f"  Val:   {len(dataset['validation']):,} rows")
    print(f"  Test:  {len(dataset['test']):,} rows")

    # 2. Train BPE tokenizer on train split
    print("\n" + "=" * 60)
    print("Step 2: Training BPE tokenizer (vocab_size={})".format(VOCAB_SIZE))
    print("=" * 60)
    train_texts = [row['text'] for row in dataset['train'] if row['text'].strip()]
    print(f"  Non-empty train texts: {len(train_texts):,}")

    tokenizer = train_tokenizer(train_texts)
    tokenizer_path = DATA_DIR / 'tokenizer.json'
    tokenizer.save(str(tokenizer_path))
    print(f"  Tokenizer saved to {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

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
    for split_name, texts in splits.items():
        print(f"\n  --- {split_name} ---")
        output_path = DATA_DIR / f'{split_name}.bin'
        n_tokens = tokenize_and_save(tokenizer, texts, output_path)
        stats[split_name] = n_tokens

    # Summary
    print("\n" + "=" * 60)
    print("Done! Summary:")
    print("=" * 60)
    for split_name, n_tokens in stats.items():
        print(f"  {split_name}: {n_tokens:,} tokens ({DATA_DIR / f'{split_name}.bin'})")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == '__main__':
    main()
