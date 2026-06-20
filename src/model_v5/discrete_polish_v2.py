"""Discrete Polish V2: Post-training ternary optimization via layer-wise reconstruction.

Terminal operation - NEVER resume training from a polished checkpoint.
V2 fixes V1's cascading problem by freezing group statistics and operating on ternary codes only.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .model import ModelV2
from .components.ternary import DEFAULT_THRESHOLD_COEFF
from .config import (
    ModelConfig,
    config_wikitext_6m,
    config_wikitext_50m,
    config_wikitext_100m,
)
from .components.ternary import TernaryLinear, TernaryQuantize
from .train import evaluate_perplexity
from .data import WikiTextDataset


CONFIG_REGISTRY = {
    'wikitext_6m': config_wikitext_6m,
    'wikitext_50m': config_wikitext_50m,
    'wikitext_100m': config_wikitext_100m,
}


def _load_checkpoint(path: str) -> dict:
    """Load checkpoint, handling multiple key formats."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        return ck['model_state_dict']
    elif isinstance(ck, dict) and 'model_state' in ck:
        return ck['model_state']
    elif isinstance(ck, dict) and any(
        k.startswith('embedding') or k.startswith('layers') for k in ck.keys()
    ):
        return ck
    else:
        return ck


def _build_alpha_matrix(module: TernaryLinear, orig_shape: tuple) -> Optional[torch.Tensor]:
    """Build per-element alpha matrix matching _apply_alpha logic. Returns None if module.alpha is None."""
    if module.alpha is None:
        return None
    numel = orig_shape[0] * orig_shape[1]
    if module.alpha_group_size is None:
        return torch.full(orig_shape, module.alpha.item(), dtype=torch.float32)
    flat = torch.ones(numel, dtype=torch.float32)
    ags = module.alpha_group_size
    pad_len = (ags - numel % ags) % ags
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))
    grouped = flat.reshape(-1, ags)
    scaled = grouped * module.alpha.data.float().unsqueeze(-1)
    return scaled.reshape(-1)[:numel].reshape(orig_shape)


def collect_calibration_data(
    model: ModelV2,
    config: ModelConfig,
    data_dir: str,
    n_calib_batches: int = 8,
    batch_size: int = 16,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """Collect activation data for layer-wise reconstruction.

    Uses training set for calibration. Registers forward hooks on all TernaryLinear modules
    to capture activations, then removes hooks after collection.
    """
    model.eval()
    model.to(device)

    dataset = WikiTextDataset(split='train', seq_len=config.seq_len, data_dir=data_dir)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    calibration_data = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            x = input[0]
            if x.dim() == 3:
                b, s, f = x.shape
                x = x.reshape(b * s, f)
            if name not in calibration_data:
                calibration_data[name] = []
            calibration_data[name].append(x.detach().float().cpu())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    print(f"Collecting calibration data from {n_calib_batches} batches...")
    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(loader):
            if batch_idx >= n_calib_batches:
                break
            input_ids = input_ids.to(device)
            _ = model(input_ids)
            if (batch_idx + 1) % max(1, n_calib_batches // 2) == 0:
                print(f"  Batch {batch_idx + 1}/{n_calib_batches}")

    for handle in handles:
        handle.remove()

    result = {}
    for name, chunks in calibration_data.items():
        result[name] = torch.cat(chunks, dim=0)
        print(f"  {name}: shape {result[name].shape}")

    return result


def freeze_quantization_state(module: TernaryLinear) -> Dict[str, torch.Tensor]:
    """Freeze quantization state by extracting ternary codes and scales.

    Mirrors TernaryQuantize.forward() exactly to ensure consistency.
    Returns dict with q_codes, scales_flat, alpha_mat, and metadata.
    """
    weight = module.weight.data.float()
    group_size = module.group_size
    threshold_coeff = getattr(module, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF)
    orig_shape = weight.shape

    flat = weight.reshape(-1)
    numel = flat.numel()
    pad_len = (group_size - numel % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    grouped = flat.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    threshold = threshold_coeff * normalized.abs().mean(dim=-1, keepdim=True)
    q_codes = normalized.sign() * (normalized.abs() > threshold).float()

    scale_flat = scales.expand_as(grouped).reshape(-1)[:numel].reshape(orig_shape)
    q_flat = q_codes.reshape(-1)[:numel].reshape(orig_shape)
    alpha_mat = _build_alpha_matrix(module, orig_shape)

    return {
        'q_codes': q_flat,
        'scales_flat': scale_flat,
        'alpha_mat': alpha_mat,
        'orig_shape': orig_shape,
        'group_size': group_size,
    }


def compute_effective_weight(frozen: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute effective weight: q_codes * scales * alpha (if present)."""
    w = frozen['q_codes'] * frozen['scales_flat']
    if frozen['alpha_mat'] is not None:
        w = w * frozen['alpha_mat']
    return w


def compute_hessian_diagonal(X: torch.Tensor) -> torch.Tensor:
    """Compute Hessian diagonal: sum of X^2 along calibration dimension."""
    return (X ** 2).sum(dim=0)


def score_all_flips(
    q_current: torch.Tensor,
    scales_flat: torch.Tensor,
    alpha_mat: Optional[torch.Tensor],
    G: torch.Tensor,
    H_diag: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score all possible single-weight flips vectorized.

    Returns (best_scores, best_targets) both shape (out, in).
    best_scores[o,i] = minimum score for position (o,i)
    best_targets[o,i] = target ternary value (-1, 0, or 1) achieving that score
    """
    targets = torch.tensor([-1.0, 0.0, 1.0], device=q_current.device)
    best_scores = torch.full_like(q_current, float('inf'))
    best_targets = torch.zeros_like(q_current)

    for t in targets:
        delta_q = t - q_current
        delta_v = delta_q * scales_flat
        if alpha_mat is not None:
            delta_v = delta_v * alpha_mat
        score = -2.0 * delta_v * G + delta_v.pow(2) * H_diag.unsqueeze(0)

        same_mask = (delta_q == 0)
        score[same_mask] = float('inf')

        improved = score < best_scores
        best_scores[improved] = score[improved]
        best_targets[improved] = t

    return best_scores, best_targets


def encode_ternary_to_shadow(module: TernaryLinear, q_target: torch.Tensor, frozen: Dict[str, torch.Tensor]):
    """Write ternary codes back to shadow weights (module.weight).

    CRITICAL: shadow = q_target * scales_flat, NOT multiplied by alpha.
    Alpha is applied separately in forward().
    """
    shadow_new = q_target * frozen['scales_flat']
    module.weight.data.copy_(shadow_new.to(module.weight.dtype))


def polish_layer_v2(
    module: TernaryLinear,
    name: str,
    X_calib: torch.Tensor,
    max_flips: int = 50,
    device: str = 'cuda',
) -> Dict:
    """Optimize single TernaryLinear layer via greedy flip search.

    Uses cached Hessian diagonal for O(1) flip cost per iteration.
    Returns stats dict with flips applied and reconstruction loss improvement.
    """
    X_calib = X_calib.to(device).float()
    W_shadow = module.weight.data.float()

    frozen = freeze_quantization_state(module)
    for key in frozen:
        if isinstance(frozen[key], torch.Tensor):
            frozen[key] = frozen[key].to(device)

    W_q_eff = compute_effective_weight(frozen)

    R = (W_shadow - W_q_eff) @ X_calib.t()
    initial_loss = (R ** 2).sum().item()

    H_diag = compute_hessian_diagonal(X_calib)

    q_current = frozen['q_codes'].clone()
    G = R @ X_calib
    best_scores, best_targets = score_all_flips(
        q_current, frozen['scales_flat'], frozen['alpha_mat'], G, H_diag
    )

    flips_applied = 0
    for flip_idx in range(max_flips):
        flat_idx = torch.argmin(best_scores.view(-1))
        best_score = best_scores.view(-1)[flat_idx].item()

        if best_score >= 0:
            break

        out_idx = flat_idx // q_current.shape[1]
        in_idx = flat_idx % q_current.shape[1]

        old_q = q_current[out_idx, in_idx].item()
        new_q = best_targets[out_idx, in_idx].item()

        delta_q = new_q - old_q
        delta_v = delta_q * frozen['scales_flat'][out_idx, in_idx]
        if frozen['alpha_mat'] is not None:
            delta_v = delta_v * frozen['alpha_mat'][out_idx, in_idx]

        R[out_idx, :] -= delta_v * X_calib[:, in_idx]
        q_current[out_idx, in_idx] = new_q

        G_row_new = R[out_idx:out_idx + 1, :] @ X_calib
        G[out_idx, :] = G_row_new.squeeze(0)

        best_scores[out_idx, :] = float('inf')

        for t in [-1.0, 0.0, 1.0]:
            delta_q_new = t - q_current[out_idx, :]
            delta_v_new = delta_q_new * frozen['scales_flat'][out_idx, :]
            if frozen['alpha_mat'] is not None:
                delta_v_new = delta_v_new * frozen['alpha_mat'][out_idx, :]
            score_new = -2.0 * delta_v_new * G[out_idx, :] + delta_v_new.pow(2) * H_diag

            same_mask = (delta_q_new == 0)
            score_new[same_mask] = float('inf')

            improved = score_new < best_scores[out_idx, :]
            best_scores[out_idx, improved] = score_new[improved]
            best_targets[out_idx, improved] = t

        flips_applied += 1

    final_loss = (R ** 2).sum().item()
    improvement = initial_loss - final_loss
    improvement_pct = 100.0 * improvement / initial_loss if initial_loss > 0 else 0

    encode_ternary_to_shadow(module, q_current, frozen)

    del X_calib, W_shadow, R, G, H_diag, frozen
    torch.cuda.empty_cache()

    return {
        'layer': name,
        'flips': flips_applied,
        'loss_before': initial_loss,
        'loss_after': final_loss,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
    }


def discrete_polish_v2(
    model: ModelV2,
    config: ModelConfig,
    device: str = 'cuda',
    data_dir: str = 'data/wikitext103',
    n_calib_batches: int = 8,
    batch_size: int = 16,
    max_flips_per_layer: int = 50,
    eval_batch_size: int = 16,
) -> Dict:
    """Post-training ternary optimization via layer-wise greedy flip search.

    WARNING: Terminal operation. Never resume training from a polished checkpoint.

    Args:
        model: Trained ModelV2 with ternary weights
        config: Model config
        device: Device for computation
        data_dir: Path to WikiText data
        n_calib_batches: Number of calibration batches (from training set)
        batch_size: Batch size for calibration
        max_flips_per_layer: Maximum flips per layer
        eval_batch_size: Batch size for perplexity evaluation

    Returns:
        Dict with optimization results
    """
    model.eval()
    model.to(device)

    for m in model.modules():
        if isinstance(m, TernaryLinear):
            m.ste_type = 'standard'

    print(f"\nDiscrete Polish V2")
    print(f"  max_flips_per_layer={max_flips_per_layer}")
    print(f"  n_calib_batches={n_calib_batches}")

    print("\nEvaluating baseline PPL...")
    baseline_ppl = evaluate_perplexity(
        model, config, split='val', batch_size=eval_batch_size,
        device=device, data_dir=data_dir
    )
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    calib_data = collect_calibration_data(
        model, config, data_dir=data_dir, n_calib_batches=n_calib_batches,
        batch_size=batch_size, device=device
    )

    layer_stats = []
    total_flips = 0
    start_time = time.time()

    print("\nOptimizing layers...")
    for name, module in model.named_modules():
        if not isinstance(module, TernaryLinear):
            continue

        if name not in calib_data:
            print(f"  {name}: no calibration data, skipping")
            continue

        X_calib = calib_data[name]
        stats = polish_layer_v2(
            module, name, X_calib, max_flips=max_flips_per_layer, device=device
        )

        layer_stats.append(stats)
        total_flips += stats['flips']

        print(f"  {name}: {stats['flips']} flips, loss {stats['loss_before']:.4f} -> {stats['loss_after']:.4f} "
              f"({stats['improvement_pct']:.2f}%)")

    del calib_data
    torch.cuda.empty_cache()

    print("\nEvaluating final PPL...")
    final_ppl = evaluate_perplexity(
        model, config, split='val', batch_size=eval_batch_size,
        device=device, data_dir=data_dir
    )
    print(f"  Final PPL: {final_ppl:.2f}")

    elapsed = time.time() - start_time
    total_improvement = baseline_ppl - final_ppl
    pct_improvement = 100.0 * total_improvement / baseline_ppl if baseline_ppl > 0 else 0

    print(f"\n{'='*60}")
    print(f"DISCRETE POLISH V2 RESULTS")
    print(f"{'='*60}")
    print(f"Baseline PPL:     {baseline_ppl:.2f}")
    print(f"Final PPL:        {final_ppl:.2f}")
    print(f"PPL improvement:  {total_improvement:.2f} ({pct_improvement:.2f}%)")
    print(f"Total flips:      {total_flips}")
    print(f"Layers optimized: {len(layer_stats)}")
    print(f"Time:             {elapsed:.1f}s")
    print(f"{'='*60}")

    return {
        'baseline_ppl': baseline_ppl,
        'final_ppl': final_ppl,
        'ppl_improvement': total_improvement,
        'pct_improvement': pct_improvement,
        'total_flips': total_flips,
        'n_layers': len(layer_stats),
        'elapsed': elapsed,
        'layer_stats': layer_stats,
        'n_calib_batches': n_calib_batches,
        'max_flips_per_layer': max_flips_per_layer,
    }


def polish_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config_name: str = 'wikitext_6m',
    device: str = 'cuda',
    data_dir: str = 'data/wikitext103',
    n_calib_batches: int = 8,
    batch_size: int = 16,
    max_flips_per_layer: int = 50,
    eval_batch_size: int = 16,
    use_ternary: bool = True,
    use_ssd: bool = True,
    use_routing: bool = True,
) -> Dict:
    """Load checkpoint, apply discrete polish V2, save result."""

    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = _load_checkpoint(checkpoint_path)

    config = CONFIG_REGISTRY[config_name]()
    config.use_ternary = use_ternary
    config.use_ssd = use_ssd
    config.use_routing = use_routing
    config.use_torch_compile = False

    model = ModelV2(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys")

    model.to(device)
    model.convert_ternary_to_bf16()

    n_ternary = sum(
        1 for m in model.modules() if isinstance(m, TernaryLinear)
    )
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{n_ternary} TernaryLinear layers")

    result = discrete_polish_v2(
        model, config, device=device, data_dir=data_dir,
        n_calib_batches=n_calib_batches, batch_size=batch_size,
        max_flips_per_layer=max_flips_per_layer, eval_batch_size=eval_batch_size,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'config_name': config_name,
            'polish_metadata': result,
        }, output_path)
        print(f"\nSaved polished checkpoint to {output_path}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discrete Polish V2: post-training ternary optimization'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained ternary checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for polished checkpoint')
    parser.add_argument('--config', type=str, default='wikitext_6m',
                       choices=list(CONFIG_REGISTRY.keys()),
                       help='Config name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-dir', type=str, default='data/wikitext103')
    parser.add_argument('--n-calib-batches', type=int, default=8,
                       help='Number of calibration batches')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for calibration')
    parser.add_argument('--max-flips', type=int, default=50,
                       help='Maximum flips per layer')
    parser.add_argument('--eval-batch-size', type=int, default=16,
                       help='Batch size for perplexity evaluation')
    parser.add_argument('--no-ternary', action='store_true')
    parser.add_argument('--no-ssd', action='store_true')
    parser.add_argument('--no-routing', action='store_true')

    args = parser.parse_args()

    polish_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_name=args.config,
        device=args.device,
        data_dir=args.data_dir,
        n_calib_batches=args.n_calib_batches,
        batch_size=args.batch_size,
        max_flips_per_layer=args.max_flips,
        eval_batch_size=args.eval_batch_size,
        use_ternary=not args.no_ternary,
        use_ssd=not args.no_ssd,
        use_routing=not args.no_routing,
    )
