import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional, List


class CUDAPrefetcher:
    """Prefetches next batch to GPU on a separate CUDA stream.

    F3 CUDA-STREAM-ASYNC-PREFETCH: Overlaps data transfer with GPU computation
    by prefetching the next batch while the current batch is being processed.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.next_batch = None
        self._iter = None

    def __iter__(self):
        self._iter = iter(self.loader)
        self._prefetch()
        return self

    def _prefetch(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                if isinstance(batch, dict):
                    self.next_batch = {
                        k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                elif isinstance(batch, (tuple, list)):
                    self.next_batch = type(batch)(
                        v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                        for v in batch
                    )
                else:
                    self.next_batch = batch.to(self.device, non_blocking=True)
        else:
            self.next_batch = batch

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._prefetch()
        return batch

    def __len__(self):
        return len(self.loader)


class PackedDataset(Dataset):
    """Packs multiple sequences into fixed-length packs with document boundary markers.

    F6 SEQUENCE-PACKING: Concatenates multiple short sequences into fixed-length packs
    to eliminate padding waste. For SSMs, boundary masks indicate where hidden state
    should reset at document boundaries within a pack.
    """

    def __init__(self, token_ids: np.ndarray, seq_len: int,
                 doc_boundaries: Optional[List[int]] = None, eos_token_id: Optional[int] = None):
        """
        Args:
            token_ids: flat array of token IDs (pre-tokenized), dtype uint16 or int64
            seq_len: target pack length
            doc_boundaries: indices in token_ids where documents start. If None and eos_token_id
                is provided, auto-detect boundaries from EOS token positions.
            eos_token_id: token ID marking end of document. If provided and doc_boundaries is None,
                will scan token_ids for EOS positions to build boundaries.
        """
        self.seq_len = seq_len
        self.token_ids = token_ids
        self.packs = []
        self.boundary_masks = []

        if doc_boundaries is None and eos_token_id is not None:
            eos_positions = np.where(token_ids == eos_token_id)[0]
            doc_boundaries = list(eos_positions + 1)
            doc_boundaries = [b for b in doc_boundaries if b < len(token_ids)]
            if len(doc_boundaries) > 0:
                print(f"[PACKING] Auto-detected {len(doc_boundaries)} document boundaries from EOS token")

        if doc_boundaries is None or len(doc_boundaries) == 0:
            n_packs = len(token_ids) // seq_len
            for i in range(n_packs):
                start = i * seq_len
                self.packs.append(start)
                self.boundary_masks.append(np.zeros(seq_len, dtype=np.bool_))
        else:
            pos = 0
            while pos + seq_len <= len(token_ids):
                self.packs.append(pos)
                mask = np.zeros(seq_len, dtype=np.bool_)
                for b in doc_boundaries:
                    offset = b - pos
                    if 0 < offset < seq_len:
                        mask[offset] = True
                pos += seq_len
                self.boundary_masks.append(mask)

            packing_efficiency = (len(self.packs) * seq_len) / len(token_ids)
            n_boundaries_total = sum(m.sum() for m in self.boundary_masks)
            print(f"[PACKING] Loaded {len(token_ids):,} tokens | Packed into {len(self.packs)} sequences")
            print(f"[PACKING] Packing efficiency: {packing_efficiency:.1%} | Boundaries: {n_boundaries_total}")

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx):
        start = self.packs[idx]
        tokens = torch.from_numpy(
            self.token_ids[start:start + self.seq_len].astype(np.int64)
        )
        boundary_mask = torch.from_numpy(self.boundary_masks[idx])
        return tokens, boundary_mask


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
    Tests retrieval capability - pure SSM should struggle, hybrid should work.
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
                 data_dir: str = None, ref_losses_path: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'wikitext103'
        else:
            data_dir = Path(data_dir)

        bin_path = data_dir / f'{split}.bin'
        if not bin_path.exists():
            raise FileNotFoundError(
                f"{bin_path} not found. Run: python -m model_v5.prepare_wikitext"
            )

        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_sequences = (len(self.data) - 1) // seq_len

        self.ref_losses = None
        if ref_losses_path is not None:
            ref_path = Path(ref_losses_path)
            if ref_path.exists():
                self.ref_losses = np.memmap(str(ref_path), dtype=np.float16, mode='r')

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        if self.ref_losses is not None:
            ref_chunk = self.ref_losses[start + 1:start + self.seq_len + 1].astype(np.float32)
            ref = torch.from_numpy(ref_chunk.copy())
            return x, y, ref

        return x, y

    def __getstate__(self):
        """Support pickling for Windows multiprocessing spawn context."""
        state = self.__dict__.copy()
        state['_mmap_path'] = str(self.data.filename)
        state['_mmap_dtype'] = str(self.data.dtype)
        del state['data']
        if getattr(self, 'ref_losses', None) is not None:
            state['_ref_path'] = str(self.ref_losses.filename)
            state['_ref_dtype'] = str(self.ref_losses.dtype)
            del state['ref_losses']
        return state

    def __setstate__(self, state):
        path = state.pop('_mmap_path')
        dtype = np.dtype(state.pop('_mmap_dtype'))
        ref_path = state.pop('_ref_path', None)
        ref_dtype = state.pop('_ref_dtype', None)
        self.__dict__.update(state)
        self.data = np.memmap(path, dtype=dtype, mode='r')
        if ref_path is not None:
            self.ref_losses = np.memmap(ref_path, dtype=np.dtype(ref_dtype), mode='r')
        else:
            self.ref_losses = None


class MemmapDataset(Dataset):
    """Memory-mapped binary dataset for zero-copy data loading.

    R4-A1-F1 MMAP-BINARY-DATASET: Pre-tokenized dataset loaded via mmap
    eliminates tokenization overhead during training.

    Expects:
        - {data_path}.bin: uint16/uint32 mmap array
        - {data_path}.meta: JSON metadata
    """

    def __init__(self, data_path: str, seq_len: int = 256):
        """Initialize MemmapDataset.

        Args:
            data_path: Path to .bin file (without extension) or full path
            seq_len: Sequence length for training
        """
        # Handle both "path/to/train" and "path/to/train.bin"
        data_path = Path(data_path)
        if data_path.suffix == '.bin':
            bin_path = data_path
            meta_path = data_path.with_suffix('.meta')
        else:
            bin_path = data_path.with_suffix('.bin')
            meta_path = data_path.with_suffix('.meta')

        if not bin_path.exists():
            raise FileNotFoundError(
                f"{bin_path} not found. Run: python -m src.model_v5.prepare_binary_dataset"
            )

        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Metadata required for mmap dataset."
            )

        # Load metadata
        with open(meta_path) as f:
            self.meta = json.load(f)

        # Determine dtype from metadata
        dtype_str = self.meta.get('dtype', 'uint16')
        if 'uint32' in dtype_str:
            dtype = np.uint32
        else:
            dtype = np.uint16

        # Memory-map the binary file
        self.data = np.memmap(bin_path, dtype=dtype, mode='r')
        self.seq_len = seq_len
        self.n_sequences = (len(self.data) - 1) // seq_len

        print(f"[MMAP] Loaded {bin_path}")
        print(f"  Tokens: {len(self.data):,} | Sequences: {self.n_sequences:,} | seq_len: {seq_len}")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        """Get training sequence.

        Returns:
            Tuple of (input_ids, target_ids)
        """
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    def __getstate__(self):
        """Support pickling for Windows multiprocessing spawn context."""
        state = self.__dict__.copy()
        state['_mmap_path'] = str(self.data.filename)
        state['_mmap_dtype'] = str(self.data.dtype)
        del state['data']
        return state

    def __setstate__(self, state):
        path = state.pop('_mmap_path')
        dtype = np.dtype(state.pop('_mmap_dtype'))
        self.__dict__.update(state)
        self.data = np.memmap(path, dtype=dtype, mode='r')


class BookDataset(Dataset):
    """Book-derived training data using Source-Signal pipeline.

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


class StreamingDataset:
    """N parallel continuous text streams for stateful SSM training.

    Instead of random-access chunks, provides N continuous streams that
    the SSM processes with persistent state. Each stream reads a different
    section of the token file.

    This maximizes SSM state utilization - the model never "forgets" because
    state is never reset between batches.

    P4: STATEFUL-STREAMING-TRAINING
    """

    def __init__(self, bin_path: str, num_streams: int, seq_len: int):
        """
        Args:
            bin_path: Path to .bin file with pre-tokenized tokens (uint16)
            num_streams: Number of parallel streams
            seq_len: Sequence length per batch
        """
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.num_streams = num_streams
        self.seq_len = seq_len

        # Divide tokens evenly among streams
        total = len(self.tokens)
        self.stream_len = total // num_streams
        self.stream_offsets = [i * self.stream_len for i in range(num_streams)]
        self.positions = list(self.stream_offsets)

    def get_batch(self) -> torch.Tensor:
        """Get next batch: one chunk per stream.

        Returns:
            tensor [num_streams, seq_len] of token IDs (int64)
        """
        batch = []
        for i in range(self.num_streams):
            pos = self.positions[i]
            end = pos + self.seq_len

            # Wrap around if needed (continue from stream start)
            if end > self.stream_offsets[i] + self.stream_len:
                pos = self.stream_offsets[i]
                end = pos + self.seq_len

            chunk = self.tokens[pos:end].astype(np.int64)
            batch.append(torch.from_numpy(chunk))
            self.positions[i] = end

        return torch.stack(batch)

    def reset(self):
        """Reset all stream positions to start."""
        self.positions = list(self.stream_offsets)


class CurriculumSampler(Sampler):
    """Sorts chunks by ascending difficulty (easy-first) using precomputed ref losses."""

    def __init__(self, ref_losses_path: str, seq_len: int):
        ref_path = Path(ref_losses_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"ref_losses not found: {ref_losses_path}")

        ref_losses = np.memmap(str(ref_path), dtype=np.float16, mode='r')
        n_tokens = len(ref_losses)
        self.n_sequences = (n_tokens - 1) // seq_len

        chunk_difficulties = np.zeros(self.n_sequences, dtype=np.float32)
        for idx in range(self.n_sequences):
            start = idx * seq_len + 1
            end = start + seq_len
            chunk_difficulties[idx] = ref_losses[start:end].astype(np.float32).mean()

        self.sorted_indices = np.argsort(chunk_difficulties).tolist()

        easy_diff = chunk_difficulties[self.sorted_indices[0]]
        hard_diff = chunk_difficulties[self.sorted_indices[-1]]
        median_diff = np.median(chunk_difficulties)
        print(f"CurriculumSampler: {self.n_sequences} chunks sorted by difficulty")
        print(f"  Easiest: {easy_diff:.4f}, Median: {median_diff:.4f}, Hardest: {hard_diff:.4f}")

    def __len__(self):
        return self.n_sequences

    def __iter__(self):
        return iter(self.sorted_indices)


def sort_documents_by_length(token_ids: np.ndarray, doc_boundaries: list, descending: bool = True) -> tuple:
    """Sort documents by length (LPT - Longest Processing Time First).

    M1 PIPELINE-OPTIMAL-SSM-SCHEDULE: Reorders documents so longer documents
    come first, improving SSM state utilization and reducing pipeline idle time.

    Returns:
        (sorted_token_ids, sorted_boundaries) where documents are reordered.
    """
    doc_lengths = []
    for i in range(len(doc_boundaries)):
        start = doc_boundaries[i]
        end = doc_boundaries[i + 1] if i + 1 < len(doc_boundaries) else len(token_ids)
        doc_lengths.append((i, end - start))

    doc_lengths.sort(key=lambda x: x[1], reverse=descending)

    sorted_tokens = []
    sorted_boundaries = [0]
    for doc_idx, length in doc_lengths:
        start = doc_boundaries[doc_idx]
        end = doc_boundaries[doc_idx + 1] if doc_idx + 1 < len(doc_boundaries) else len(token_ids)
        sorted_tokens.append(token_ids[start:end])
        sorted_boundaries.append(sorted_boundaries[-1] + length)

    return np.concatenate(sorted_tokens), sorted_boundaries[:-1]


def build_document_index(bin_path: str, eos_token_id: int) -> np.ndarray:
    """Scan a memmap .bin file and find document boundaries (EOS token positions).

    M2 STRATIFIED-CONTIGUOUS-MMAP: Pre-computes document boundaries by scanning
    for EOS tokens, enabling efficient stratified sampling and document-aligned packing.

    Returns:
        Array of indices where documents start.
        Saves index to bin_path.replace('.bin', '.idx') for reuse.
    """
    idx_path = bin_path.replace('.bin', '.idx')
    if os.path.exists(idx_path):
        return np.load(idx_path)

    tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
    eos_positions = np.where(tokens == eos_token_id)[0]
    doc_starts = np.concatenate([[0], eos_positions + 1])
    doc_starts = doc_starts[doc_starts < len(tokens)]

    np.save(idx_path, doc_starts)
    return doc_starts


def precompute_ternary_embeddings(embedding_weight: torch.Tensor, token_ids: np.ndarray,
                                   cache_path: str = None) -> dict:
    """Pre-compute ternary embedding lookups for all tokens in dataset.

    P2 TERNARY-ENCODED-DATA-FORMAT: Turns first embedding layer into integer
    accumulation by pre-computing ternary quantized embeddings for all tokens.

    Args:
        embedding_weight: The model's embedding weight matrix [vocab_size, d_model]
        token_ids: Array of token IDs
        cache_path: Optional path to save/load cache

    Returns:
        Dict mapping token ID to pre-computed ternary embedding vector.
    """
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    from .quantization import quantize_weights_ste
    ternary_emb = quantize_weights_ste(embedding_weight.detach())

    unique_ids = np.unique(token_ids)
    lookup = {}
    for uid in unique_ids:
        lookup[int(uid)] = ternary_emb[uid]

    if cache_path:
        torch.save(lookup, cache_path)

    return lookup


def get_dataloader(dataset_type: str, vocab_size: int, seq_len: int,
                   batch_size: int, num_samples: int = 10000,
                   num_workers: int = 4, pin_memory: bool = True,
                   persistent_workers: bool = True, prefetch_factor: int = 4,
                   sampler=None, use_sequence_packing: bool = None,
                   force_disable_packing: bool = False, **kwargs):
    """Create a DataLoader for experiments.

    Args:
        num_workers: Number of data loading workers (0 = main process only).
        pin_memory: Pin memory for faster CPU-GPU transfer.
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0).
        prefetch_factor: Number of batches to prefetch per worker (requires num_workers > 0).
        use_sequence_packing: Enable F6 sequence packing (eliminates padding waste).
            If None (default), auto-enable for wikitext, disable for others.
        force_disable_packing: Debug override to disable packing even when normally enabled.
    """
    if use_sequence_packing is None:
        use_sequence_packing = (dataset_type == 'wikitext')

    if use_sequence_packing and not force_disable_packing:
        if dataset_type == 'wikitext':
            split = kwargs.get('split', 'train')
            data_dir = kwargs.get('data_dir', None)
            if data_dir is None:
                data_dir = Path(__file__).parent.parent.parent / 'data' / 'wikitext103'
            else:
                data_dir = Path(data_dir)

            bin_path = data_dir / f'{split}.bin'
            if not bin_path.exists():
                raise FileNotFoundError(
                    f"{bin_path} not found. Run: python -m model_v5.prepare_wikitext"
                )

            token_ids = np.memmap(bin_path, dtype=np.uint16, mode='r')
            eos_token_id = kwargs.get('eos_token_id', 50256)
            dataset = PackedDataset(token_ids, seq_len, doc_boundaries=None, eos_token_id=eos_token_id)
            print(f"[PACKING] Sequence packing ENABLED for wikitext")
        else:
            raise ValueError(f"Sequence packing not supported for dataset type '{dataset_type}'")
    else:
        if dataset_type == 'synthetic':
            dataset = SyntheticTextDataset(vocab_size, seq_len, num_samples)
        elif dataset_type == 'copy':
            dataset = CopyTaskDataset(vocab_size, seq_len, num_samples=num_samples, **kwargs)
        elif dataset_type == 'wikitext':
            split = kwargs.get('split', 'train')
            data_dir = kwargs.get('data_dir', None)
            ref_losses_path = kwargs.get('ref_losses_path', None)

            # R4-A1-F1: Try MMAP binary dataset first (zero tokenization overhead)
            if data_dir is None:
                data_dir = Path(__file__).parent.parent.parent / 'data' / 'wikitext103'
            else:
                data_dir = Path(data_dir)

            # Check for pre-tokenized binary (e.g., train_packed.bin)
            mmap_path = data_dir / f'{split}_packed'
            if (mmap_path.with_suffix('.bin').exists() and
                mmap_path.with_suffix('.meta').exists()):
                print(f"[DATA] Using pre-tokenized MMAP dataset: {mmap_path}.bin")
                dataset = MemmapDataset(data_path=str(mmap_path), seq_len=seq_len)
            else:
                # Fallback to WikiTextDataset (on-the-fly processing)
                print(f"[DATA] MMAP dataset not found, using WikiTextDataset (on-the-fly)")
                print(f"  (Run prepare_binary_dataset to enable MMAP path)")
                dataset = WikiTextDataset(split=split, seq_len=seq_len, data_dir=data_dir,
                                         ref_losses_path=ref_losses_path)
        elif dataset_type == 'book':
            data_dir = kwargs.get('data_dir')
            if data_dir is None:
                raise ValueError("data_dir required for 'book' dataset type")
            dataset = BookDataset(data_dir=data_dir, seq_len=seq_len)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    if use_sequence_packing and not force_disable_packing:
        def collate_packed(batch):
            tokens_list = [item[0] for item in batch]
            masks_list = [item[1] for item in batch]
            return {
                'input_ids': torch.stack(tokens_list),
                'boundary_mask': torch.stack(masks_list),
            }
        collate_fn = collate_packed
    else:
        collate_fn = None
        if dataset_type == 'wikitext':
            print(f"[PACKING] Sequence packing DISABLED")

    dl_kwargs = {'batch_size': batch_size, 'drop_last': True}
    if sampler is not None:
        dl_kwargs['sampler'] = sampler
        dl_kwargs['shuffle'] = False
    else:
        dl_kwargs['shuffle'] = True

    import sys
    if sys.platform == "win32" and num_workers > 0:
        print(
            f"[DATALOADER-H] Windows detected: num_workers {num_workers}->0 "
            f"(dynamic module not picklable under spawn; Linux runs get full async benefit)"
        )
        num_workers = 0
        persistent_workers = False

    if pin_memory:
        dl_kwargs['pin_memory'] = True
    if num_workers > 0:
        dl_kwargs['num_workers'] = num_workers
        if persistent_workers:
            dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = prefetch_factor
    if collate_fn is not None:
        dl_kwargs['collate_fn'] = collate_fn

    if num_workers > 0:
        trinity_features = []
        if num_workers >= 4:
            trinity_features.append(f"num_workers={num_workers}")
        if pin_memory:
            trinity_features.append("pin_memory=True")
        if persistent_workers:
            trinity_features.append("persistent_workers=True")

        print(f"[DATALOADER TRINITY] Active optimizations: {', '.join(trinity_features)}")
        if len(trinity_features) == 3 and num_workers >= 4:
            print(f"[DATALOADER TRINITY] [OK] Full Trinity enabled - expect 1.25-1.30x data loading speedup")
    else:
        print(f"[DATALOADER TRINITY] WARNING: num_workers=0 (single-process mode) - Trinity DISABLED")

    return DataLoader(dataset, **dl_kwargs)
