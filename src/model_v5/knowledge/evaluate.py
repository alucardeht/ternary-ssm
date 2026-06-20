import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from typing import Dict, Optional


def evaluate_knowledge(
    model: nn.Module,
    domain_data_dir: str,
    baseline_data_dir: Optional[str] = None,
    device: str = 'cuda',
    batch_size: int = 32,
    seq_len: int = 512,
    max_batches: int = 50,
) -> Dict:
    """Evaluate knowledge injection quality via perplexity.

    Computes perplexity on domain-specific data and optionally on baseline data
    for comparison.

    Args:
        model: Model to evaluate
        domain_data_dir: Directory with domain-specific BookDataset
        baseline_data_dir: Optional directory with baseline BookDataset for comparison
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        seq_len: Sequence length (must match dataset)
        max_batches: Max batches to evaluate (for speed)

    Returns:
        Dict with domain_ppl, baseline_ppl (if provided), training_time
    """
    model.to(device)
    model.eval()

    domain_ppl = _compute_perplexity(model, domain_data_dir, device, batch_size, seq_len, max_batches)

    results = {'domain_ppl': domain_ppl}

    if baseline_data_dir:
        baseline_ppl = _compute_perplexity(model, baseline_data_dir, device, batch_size, seq_len, max_batches)
        results['baseline_ppl'] = baseline_ppl

    return results


def _compute_perplexity(
    model: nn.Module,
    data_dir: str,
    device: str,
    batch_size: int,
    seq_len: int,
    max_batches: int,
) -> float:
    """Compute perplexity on a dataset.

    Args:
        model: Model to evaluate
        data_dir: Directory with BookDataset
        device: Device to evaluate on
        batch_size: Batch size
        seq_len: Sequence length
        max_batches: Max batches to evaluate

    Returns:
        Perplexity value
    """
    from ..data_pipeline.dataset import BookDataset

    try:
        dataset = BookDataset(data_dir=data_dir, seq_len=seq_len)
    except FileNotFoundError as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        return float('inf')

    if len(dataset) == 0:
        print(f"Warning: Dataset at {data_dir} is empty")
        return float('inf')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            if i >= max_batches:
                break

            inp = inp.to(device)
            tgt = tgt.to(device)

            logits = model(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))

            total_loss += loss.item()
            total_tokens += tgt.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    return perplexity
