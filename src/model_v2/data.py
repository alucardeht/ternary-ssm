import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from pathlib import Path


class SyntheticTextDataset(Dataset):
    """Generates simple synthetic sequences for quick validation.

    Creates sequences where the model must learn simple patterns:
    - Repeating sequences (test memory)
    - Incrementing sequences (test counting)
    - Random sequences (test capacity)

    For real training, replace with TinyStories or WikiText.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        rng = random.Random(seed)
        torch.manual_seed(seed)

        self.data = []
        for i in range(num_samples):
            pattern = rng.choice(['repeat', 'increment', 'random'])
            if pattern == 'repeat':
                pattern_len = rng.randint(2, min(16, seq_len))
                base = [rng.randint(1, vocab_size - 1) for _ in range(pattern_len)]
                seq = (base * (seq_len // pattern_len + 1))[:seq_len + 1]
            elif pattern == 'increment':
                start = rng.randint(1, vocab_size - 1)
                step = rng.randint(1, 5)
                seq = [(start + j * step) % (vocab_size - 1) + 1 for j in range(seq_len + 1)]
            else:
                seq = [rng.randint(1, vocab_size - 1) for _ in range(seq_len + 1)]

            self.data.append(torch.tensor(seq[:seq_len + 1], dtype=torch.int32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx].long()
        return seq[:-1], seq[1:]  # input, target


class CopyTaskDataset(Dataset):
    """Copy task: input sequence must be reproduced in output.

    Format: [SEP] [random tokens] [SEP] [should predict: same random tokens]
    Tests retrieval capability — pure SSM should struggle, hybrid should work.
    """

    def __init__(self, vocab_size: int, seq_len: int, copy_len: int = 32,
                 num_samples: int = 10000, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.copy_len = copy_len
        self.num_samples = num_samples
        self.sep_token = 0

        torch.manual_seed(seed)
        self.data = []
        for _ in range(num_samples):
            tokens = torch.randint(1, vocab_size, (copy_len,), dtype=torch.int32)
            sep = torch.tensor([self.sep_token], dtype=torch.int32)
            full_seq = torch.cat([sep, tokens, sep, tokens])

            if len(full_seq) < seq_len + 1:
                pad = torch.zeros(seq_len + 1 - len(full_seq), dtype=torch.int32)
                full_seq = torch.cat([full_seq, pad])
            else:
                full_seq = full_seq[:seq_len + 1]

            self.data.append(full_seq)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx].long()
        return seq[:-1], seq[1:]


class WikiTextDataset(Dataset):
    """Pre-tokenized WikiText-103 dataset using memory-mapped .bin files.

    Expects data prepared by prepare_wikitext.py in data/wikitext103/.
    Chunks the token stream into sequences of (seq_len + 1) for next-token prediction.
    """

    def __init__(self, split: str = 'train', seq_len: int = 256,
                 data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'wikitext103'
        else:
            data_dir = Path(data_dir)

        bin_path = data_dir / f'{split}.bin'
        if not bin_path.exists():
            raise FileNotFoundError(
                f"{bin_path} not found. Run: python -m model_v2.prepare_wikitext"
            )

        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_sequences = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class BookDataset(Dataset):
    """Book-derived training data using Source→Signal pipeline.

    Expects data directory with:
    - examples.jsonl: One JSON per example
    - tokens.bin: Pre-tokenized binary
    - meta.json: Metadata
    """

    def __init__(self, data_dir: str, seq_len: int = 512):
        from .data_pipeline.dataset import BookDataset as _BookDataset
        self._dataset = _BookDataset(data_dir=data_dir, seq_len=seq_len)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


def get_dataloader(dataset_type: str, vocab_size: int, seq_len: int,
                   batch_size: int, num_samples: int = 10000,
                   num_workers: int = 0, pin_memory: bool = False,
                   persistent_workers: bool = False, **kwargs):
    """Create a DataLoader for experiments.

    Args:
        num_workers: Number of data loading workers (0 = main process only).
        pin_memory: Pin memory for faster CPU→GPU transfer.
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0).
    """
    if dataset_type == 'synthetic':
        dataset = SyntheticTextDataset(vocab_size, seq_len, num_samples)
    elif dataset_type == 'copy':
        dataset = CopyTaskDataset(vocab_size, seq_len, num_samples=num_samples, **kwargs)
    elif dataset_type == 'wikitext':
        split = kwargs.get('split', 'train')
        data_dir = kwargs.get('data_dir', None)
        dataset = WikiTextDataset(split=split, seq_len=seq_len, data_dir=data_dir)
    elif dataset_type == 'book':
        data_dir = kwargs.get('data_dir')
        if data_dir is None:
            raise ValueError("data_dir required for 'book' dataset type")
        dataset = BookDataset(data_dir=data_dir, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dl_kwargs = {'batch_size': batch_size, 'shuffle': True, 'drop_last': True}
    if pin_memory:
        dl_kwargs['pin_memory'] = True
    if num_workers > 0:
        dl_kwargs['num_workers'] = num_workers
        if persistent_workers:
            dl_kwargs['persistent_workers'] = True

    return DataLoader(dataset, **dl_kwargs)
