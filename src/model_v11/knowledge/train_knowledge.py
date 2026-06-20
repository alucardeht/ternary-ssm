import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Dict

from .atm import ATMAdapter
from .t_sigsgd import TSignSGD
from .anti_forgetting import setup_anti_forgetting


def train_knowledge(
    model: nn.Module,
    data_dir: str,
    device: str = 'cuda',
    num_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    filter_ratio: float = 0.95,
    max_steps: Optional[int] = None,
    alpha_warmup_steps: int = 50,
    log_every: int = 10,
    save_path: Optional[str] = None,
    seq_len: int = 512,
    max_memory_mb: int = 10000,
    target_layers: Optional[List[str]] = None,
    skip_layers: int = 0,
    fisher_path: Optional[str] = None,
    calibration_data_dir: Optional[str] = None,
    fisher_threshold: float = 1e-6,
    use_fisher: bool = True,
    use_projection: bool = True,
    replay_data_dir: Optional[str] = None,
    replay_ratio: float = 0.3,
) -> Dict:
    """Inject knowledge from a processed book dataset into the model via ATM modules.

    Args:
        model: The model to inject knowledge into
        data_dir: Directory containing BookDataset (tokens.bin, meta.json, examples.jsonl)
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Number training epochs
        batch_size: Training batch size
        lr: Learning rate for TSignSGD optimizer
        filter_ratio: Gradient magnitude filtering (keep top 1-filter_ratio)
        max_steps: Maximum training steps (None = no limit)
        alpha_warmup_steps: Warmup steps for ATM alpha scaling
        log_every: Log metrics every N steps
        save_path: Path to save trained ATM modules (None = no save)
        seq_len: Sequence length (must match dataset)
        max_memory_mb: Max GPU memory to use (safety limit)
        target_layers: List of layer name patterns to attach ATM to (None = all TernaryLinear)
        skip_layers: Number of initial layers to skip (layer index < skip_layers not adapted)
        fisher_path: Path to Fisher diagonal checkpoint for masking
        calibration_data_dir: WikiText directory for PEGP calibration
        fisher_threshold: Fisher threshold for sleeping zero identification
        use_fisher: Whether to apply Fisher masking
        use_projection: Whether to apply gradient projection

    Returns:
        Dict with training metrics: final_loss, steps, training_time, atm_params, atm_nonzero_pct, atm_size_bytes
    """
    model.to(device)
    model.train()

    adapter = ATMAdapter()
    atm_params = adapter.attach(model, target_layers, skip_layers=skip_layers)

    if not atm_params:
        print("Warning: No ATM modules attached. No TernaryLinear layers found or matched.")
        return {
            'final_loss': float('inf'),
            'steps': 0,
            'training_time': 0.0,
            'atm_params': 0,
            'atm_nonzero_pct': 0.0,
            'atm_size_bytes': 0,
        }

    total_atm_params = sum(p.numel() for p in atm_params)
    print(f"ATM attached to {len(adapter._atm_modules)} layers | Total ATM params: {total_atm_params:,}")

    calibration_loader = None
    if calibration_data_dir is not None:
        from ..datasets import WikiTextDataset

        try:
            calibration_dataset = WikiTextDataset(
                data_dir=calibration_data_dir,
                seq_len=seq_len,
                split='validation',
            )
            calibration_loader = DataLoader(
                calibration_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Could not load calibration data: {e}")

    af_stats = setup_anti_forgetting(
        model,
        adapter,
        fisher_path=fisher_path,
        calibration_loader=calibration_loader,
        fisher_threshold=fisher_threshold,
        use_fisher=use_fisher,
        use_projection=use_projection,
        device=device,
    )

    optimizer = TSignSGD(atm_params, lr=lr, filter_ratio=filter_ratio)

    from ..data_pipeline.dataset import BookDataset

    try:
        dataset = BookDataset(data_dir=data_dir, seq_len=seq_len)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {
            'final_loss': float('inf'),
            'steps': 0,
            'training_time': 0.0,
            'atm_params': total_atm_params,
            'atm_nonzero_pct': 0.0,
            'atm_size_bytes': total_atm_params * 4,
        }

    if len(dataset) == 0:
        print("Warning: Dataset is empty.")
        return {
            'final_loss': float('inf'),
            'steps': 0,
            'training_time': 0.0,
            'atm_params': total_atm_params,
            'atm_nonzero_pct': 0.0,
            'atm_size_bytes': total_atm_params * 4,
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Data replay: mix base data to prevent ATM from disrupting base knowledge
    replay_iter = None
    replay_interval = None
    if replay_data_dir is not None and replay_ratio > 0:
        from ..datasets import WikiTextDataset as ReplayDataset
        try:
            replay_dataset = ReplayDataset(
                data_dir=replay_data_dir,
                seq_len=seq_len,
                split='train',
            )
            if len(replay_dataset) > 0:
                replay_loader = DataLoader(
                    replay_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                )
                replay_iter = iter(replay_loader)
                replay_interval = math.ceil(1.0 / replay_ratio)
                print(f"Data replay enabled: interval={replay_interval} (~{100.0/replay_interval:.0f}% replay) | Replay samples: {len(replay_dataset):,}")
            else:
                print("Warning: Replay dataset is empty, skipping data replay")
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Could not load replay data: {e}")

    step = 0
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    final_loss = float('inf')

    for epoch in range(num_epochs):
        for batch_input, batch_target in dataloader:
            # Data replay: periodically swap domain batch with replay batch
            is_replay = False
            if replay_iter is not None and step % replay_interval == 0:
                try:
                    batch_input, batch_target = next(replay_iter)
                    is_replay = True
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    batch_input, batch_target = next(replay_iter)
                    is_replay = True

            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            alpha = min(1.0, step / max(alpha_warmup_steps, 1))
            adapter.set_alpha(alpha)

            logits = model(batch_input)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch_target.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            final_loss = loss.item()

            if step % log_every == 0:
                batch_type = "replay" if is_replay else "domain"
                stats = adapter.get_stats()
                total_nonzero = sum(s['nonzero_count'] for s in stats.values())
                total_weights = sum(s['total_count'] for s in stats.values())
                nonzero_pct = 100.0 * total_nonzero / max(total_weights, 1)
                print(f"Step {step:5d} | Loss {final_loss:.4f} | Alpha {alpha:.3f} | {batch_type:6s} | ATM nonzero {total_nonzero:,}/{total_weights:,} ({nonzero_pct:.1f}%)")

            step += 1

            if max_steps and step >= max_steps:
                break

        if max_steps and step >= max_steps:
            break

    training_time = time.time() - start_time

    stats = adapter.get_stats()
    final_nonzero = sum(s['nonzero_count'] for s in stats.values())
    final_total = sum(s['total_count'] for s in stats.values())
    final_nonzero_pct = 100.0 * final_nonzero / max(final_total, 1)
    final_size_bytes = final_total * 4

    if save_path:
        from .manager import ATMManager
        ATMManager.save(
            adapter,
            save_path,
            metadata={
                'steps': step,
                'final_loss': final_loss,
                'data_dir': data_dir,
                'alpha_warmup_steps': alpha_warmup_steps,
                'batch_size': batch_size,
                'learning_rate': lr,
                'replay_data_dir': replay_data_dir,
                'replay_ratio': replay_ratio,
            }
        )

    return {
        'final_loss': final_loss,
        'steps': step,
        'training_time': training_time,
        'atm_params': total_atm_params,
        'atm_nonzero_pct': final_nonzero_pct,
        'atm_size_bytes': final_size_bytes,
    }
