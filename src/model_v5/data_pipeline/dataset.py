import json
import numpy as np
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from . import TrainingExample
from .config import PipelineConfig


def save_dataset(
    examples: List[TrainingExample],
    output_dir: str,
    tokenizer_path: str,
    seq_len: int
) -> None:
    """Save training examples to dataset directory.

    Saves:
    - examples.jsonl: One JSON per example
    - tokens.bin: Pre-tokenized binary (uint16/uint32 memmap)
    - meta.json: Metadata

    Args:
        examples: List of TrainingExample objects
        output_dir: Output directory path
        tokenizer_path: Path to tokenizer file
        seq_len: Sequence length
    """
    from tokenizers import Tokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    examples_file = output_path / 'examples.jsonl'
    with open(examples_file, 'w') as f:
        for ex in examples:
            record = {
                'text': ex.text,
                'view_type': ex.view_type,
                'source_fact': ex.source_fact,
                'difficulty': ex.difficulty,
                'info_score': ex.info_score,
            }
            f.write(json.dumps(record) + '\n')

    if tokenizer_path is None:
        # Save without tokenization
        tokens_file = output_path / "tokens.bin"
        meta = {
            'n_examples': len(examples),
            'n_tokens': 0,
            'seq_len': seq_len,
            'tokenizer': None,
        }
        with open(output_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)

    all_tokens = []
    for ex in examples:
        tokens = tokenizer.encode(ex.text).ids
        all_tokens.extend(tokens)

    token_array = np.array(all_tokens, dtype=np.uint32)
    tokens_file = output_path / 'tokens.bin'
    token_array.tofile(tokens_file)

    meta = {
        'n_examples': len(examples),
        'n_tokens': len(all_tokens),
        'seq_len': seq_len,
        'tokenizer_path': str(tokenizer_path),
        'pipeline_config': {
            'score_top_k_pct': 0.25,
            'min_example_tokens': 10,
            'max_example_tokens': 256,
        }
    }
    meta_file = output_path / 'meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)


class BookDataset(Dataset):
    """Dataset for book-derived training data.

    Expects data directory with:
    - examples.jsonl: One JSON per example
    - tokens.bin: Pre-tokenized binary
    - meta.json: Metadata
    """

    def __init__(self, data_dir: str, seq_len: int = 512):
        """Initialize BookDataset.

        Args:
            data_dir: Path to dataset directory
            seq_len: Sequence length for training
        """
        data_path = Path(data_dir)

        meta_file = data_path / 'meta.json'
        if not meta_file.exists():
            raise FileNotFoundError(f"meta.json not found in {data_dir}")

        with open(meta_file) as f:
            self.meta = json.load(f)

        tokens_file = data_path / 'tokens.bin'
        if not tokens_file.exists():
            raise FileNotFoundError(f"tokens.bin not found in {data_dir}")

        self.tokens = np.memmap(tokens_file, dtype=np.uint32, mode='r')
        self.seq_len = seq_len
        self.n_sequences = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        """Get training sequence.

        Returns:
            Tuple of (input_ids, target_ids)
        """
        import torch

        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
