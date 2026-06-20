"""
Rho-1 Selective Language Modeling (NeurIPS 2024)

Core insight: Only ~3% of pretraining tokens are truly informative.
Method: Train reference model, compute per-token excess loss, mask uninformative tokens.

excess_loss(token) = student_loss(token) - reference_loss(token)
Keep top K% tokens by excess_loss per batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .config import ModelConfig, config_wikitext_6m
from .model import ModelV2
from .data import WikiTextDataset


def config_rho1_reference(vocab_size=32000, seq_len=256) -> ModelConfig:
    """Create reference model config (full precision, no ternary/routing)."""
    config = config_wikitext_6m()
    config.vocab_size = vocab_size
    config.seq_len = seq_len
    config.use_ternary = False
    config.use_routing = False
    config.use_torch_compile = False
    return config


def train_reference_model(
    data_dir,
    save_path,
    device='cuda',
    max_steps=2000,
    batch_size=32,
    lr=6e-4,
    seq_len=256,
    vocab_size=32000,
    log_every=100,
) -> str:
    """Train reference model for computing per-token excess loss."""
    from torch.utils.data import DataLoader

    config = config_rho1_reference(vocab_size, seq_len)
    model = ModelV2(config).to(device)

    dataset = WikiTextDataset(split='train', seq_len=seq_len, data_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    criterion = nn.CrossEntropyLoss()
    model.train()

    global_step = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    for input_ids, targets in loader:
        if global_step >= max_steps:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (global_step + 1) % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {global_step + 1:6d} | Loss {loss.item():.4f} | LR {current_lr:.2e}")

        global_step += 1

    end_time.record()
    torch.cuda.synchronize()
    elapsed = start_time.elapsed_time(end_time) / 1000.0

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'config': config}, save_path)

    print(f"\nReference model training complete")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Saved to {save_path}")

    return save_path


def precompute_reference_losses(
    model_path,
    data_dir,
    output_path,
    device='cuda',
    batch_size=64,
    seq_len=256,
    vocab_size=32000,
) -> str:
    """Precompute per-token reference losses for entire training set."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.vocab_size = vocab_size
    config.seq_len = seq_len

    model = ModelV2(config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    bin_path = Path(data_dir) / 'train.bin'
    data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
    n_tokens = len(data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ref_losses = np.memmap(output_path, dtype=np.float16, mode='w+', shape=(n_tokens,))
    ref_losses[0] = 0.0

    dataset = WikiTextDataset(split='train', seq_len=seq_len, data_dir=data_dir)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss(reduction='none')
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)

            per_token = criterion(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1)
            )

            per_token = per_token.reshape(input_ids.shape[0], seq_len).float().cpu().numpy()

            for i, sample_idx in enumerate(range(batch_idx * batch_size, (batch_idx + 1) * batch_size)):
                if sample_idx >= len(dataset):
                    break
                start = sample_idx * seq_len
                ref_losses[start + 1:start + seq_len + 1] = per_token[i].astype(np.float16)

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{total_batches} batches")

    ref_losses.flush()
    print(f"\nReference losses precomputed")
    print(f"Saved to {output_path}")

    return output_path


def compute_rho1_loss(
    student_logits,
    targets,
    ref_losses,
    keep_ratio=0.5,
    vocab_size=32000,
    weighting_mode='hard_mask',
) -> Tuple[torch.Tensor, dict]:
    """Compute Rho-1 selective loss: keep top K% tokens by excess loss.

    Args:
        student_logits: (B, L, V) student model logits
        targets: (B, L) target tokens
        ref_losses: (B, L) reference model per-token losses
        keep_ratio: fraction of tokens to keep (0.0-1.0)
        vocab_size: vocabulary size
        weighting_mode: 'hard_mask' (top-K% by excess loss) or
                        'relu_soft' (ReLU soft weighting, zero hyperparameters).
                        IS variance minimization. w_i = max(0, excess_i).

    Returns:
        loss: scalar training loss
        stats: dict with kept_ratio, mean_excess, threshold
    """
    B, L, V = student_logits.shape

    per_token_loss = F.cross_entropy(
        student_logits.reshape(B * L, V),
        targets.reshape(B * L),
        reduction='none',
    )

    ref_flat = ref_losses.reshape(B * L)

    excess = per_token_loss - ref_flat

    if weighting_mode not in ('hard_mask', 'relu_soft'):
        raise ValueError(f"Unknown weighting_mode: {weighting_mode}")

    n_total = excess.shape[0]

    if weighting_mode == 'hard_mask':
        n_keep = max(1, int(n_total * keep_ratio))
        threshold = torch.topk(excess, n_keep, largest=True, sorted=False).values.min()
        mask = (excess >= threshold).float()
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        stats = {
            'kept_ratio': mask.sum().item() / n_total,
            'mean_excess': excess.mean().item(),
            'threshold': threshold.item(),
        }
    else:
        w = torch.relu(excess.detach())
        w_mean = w.mean().clamp(min=1e-8)
        w_normalized = w / w_mean
        loss = (per_token_loss * w_normalized).mean()

        n_active = (w > 0).sum().item()
        stats = {
            'kept_ratio': n_active / n_total,
            'mean_excess': excess.mean().item(),
            'threshold': 0.0,
            'mean_weight': w_normalized[w > 0].mean().item() if n_active > 0 else 0.0,
        }

    return loss, stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rho-1 Reference Model Tools')
    sub = parser.add_subparsers(dest='command')

    train_p = sub.add_parser('train', help='Train reference model')
    train_p.add_argument('--data-dir', required=True)
    train_p.add_argument('--save-path', default='.checkpoints/rho1_reference.pt')
    train_p.add_argument('--device', default='cuda')
    train_p.add_argument('--max-steps', type=int, default=2000)
    train_p.add_argument('--batch-size', type=int, default=32)
    train_p.add_argument('--seq-len', type=int, default=256)

    pre_p = sub.add_parser('precompute', help='Precompute reference losses')
    pre_p.add_argument('--model-path', required=True)
    pre_p.add_argument('--data-dir', required=True)
    pre_p.add_argument('--output-path', default='data/wikitext103/ref_losses.bin')
    pre_p.add_argument('--device', default='cuda')
    pre_p.add_argument('--batch-size', type=int, default=64)
    pre_p.add_argument('--seq-len', type=int, default=256)

    args = parser.parse_args()
    if args.command == 'train':
        train_reference_model(
            args.data_dir, args.save_path, args.device,
            args.max_steps, args.batch_size, seq_len=args.seq_len
        )
    elif args.command == 'precompute':
        precompute_reference_losses(
            args.model_path, args.data_dir, args.output_path,
            args.device, args.batch_size, seq_len=args.seq_len
        )
