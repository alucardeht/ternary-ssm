"""Discrete Polish: Post-training ternary weight optimization.

Terminal operation - NEVER resume training from a polished checkpoint.
The shadow weights are set to synthetic positions to produce desired ternary states,
not shaped by gradient descent. Optimizer state would be stale.
"""

import argparse
import torch
import torch.nn.functional as F
import time
import math
from pathlib import Path
from typing import Optional, Dict
from .components.ternary import DEFAULT_THRESHOLD_COEFF

from .model import ModelV2
from .config import (
    ModelConfig,
    config_wikitext_6m,
    config_wikitext_50m,
    config_wikitext_100m,
)
from .components.ternary import TernaryLinear
from .train import evaluate_perplexity

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


def _compute_border_info(weight_data: torch.Tensor, group_size: int = 128,
                          threshold_coeff: float = DEFAULT_THRESHOLD_COEFF):
    """Compute quantization info and identify border weights.

    Returns dict with: grouped, scales, normalized, threshold, is_nonzero,
                       distance, original_shape, pad_len, real_numel
    """
    original_shape = weight_data.shape
    flat = weight_data.reshape(-1)
    real_numel = flat.numel()

    pad_len = (group_size - real_numel % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    grouped = flat.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    threshold = threshold_coeff * normalized.abs().mean(dim=-1, keepdim=True)

    is_nonzero = normalized.abs() > threshold
    distance = (normalized.abs() - threshold).abs()

    return {
        'grouped': grouped,
        'scales': scales,
        'normalized': normalized,
        'threshold': threshold,
        'is_nonzero': is_nonzero,
        'distance': distance,
        'original_shape': original_shape,
        'pad_len': pad_len,
        'real_numel': real_numel,
    }


def _apply_flips(info: dict, border_mask: torch.Tensor, push_margin: float = 0.3):
    """Compute flipped shadow weight values for border weights.

    Rules:
    - ±1 -> 0: set shadow weight to exactly 0.0
    - 0 -> ±1: set shadow to sign * scale * (threshold + push_margin)

    Returns: flipped grouped tensor (same shape as info['grouped'])
    """
    grouped = info['grouped']
    scales = info['scales']
    threshold = info['threshold']
    is_nonzero = info['is_nonzero']

    flipped = grouped.clone()

    mask_to_zero = border_mask & is_nonzero
    flipped[mask_to_zero] = 0.0

    mask_to_nonzero = border_mask & ~is_nonzero
    sign = grouped.sign()
    sign[sign == 0] = 1.0
    target_value = sign * scales * (threshold + push_margin)
    flipped[mask_to_nonzero] = target_value[mask_to_nonzero]

    return flipped


def discrete_polish(
    model: ModelV2,
    config: ModelConfig,
    device: str = 'cuda',
    data_dir: str = 'data/wikitext103',
    epsilon: float = 0.1,
    push_margin: float = 0.3,
    sign_epsilon: float = 1e-5,
    threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
    max_rounds: int = 10,
    eval_batches: Optional[int] = None,
    batch_size: int = 16,
) -> Dict:
    """Post-training ternary weight optimization via border weight flipping.

    WARNING: Terminal operation. Never resume training from a polished checkpoint.

    Args:
        model: Trained ModelV2 with ternary weights
        config: Model config
        device: Device for evaluation
        data_dir: Path to WikiText data
        epsilon: Border distance threshold (in normalized space). Default 0.1.
        push_margin: How far past threshold to push 0->±1 flips. Default 0.3.
        sign_epsilon: Skip 0->±1 flips if |shadow| < this. Default 1e-5.
        threshold_coeff: TWN threshold coefficient. Default 0.7 (matches training).
        max_rounds: Maximum polish rounds. Default 10.
        eval_batches: Limit eval to N batches for speed (None = full eval).
        batch_size: Batch size for perplexity evaluation.

    Returns:
        Dict with polish results (baseline_ppl, final_ppl, improvement, per-layer stats)
    """
    model.eval()
    model.to(device)

    print(f"\nDiscrete Polish")
    print(f"  epsilon={epsilon}, push_margin={push_margin}, max_rounds={max_rounds}")
    print(f"  eval_batches={eval_batches or 'full'}")

    print("\nEvaluating baseline PPL...")
    baseline_ppl = evaluate_perplexity(
        model, config, split='val', batch_size=batch_size,
        device=device, data_dir=data_dir
    )
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    current_ppl = baseline_ppl
    total_flips = 0
    layer_stats = []
    start_time = time.time()

    for round_idx in range(max_rounds):
        print(f"\n--- Round {round_idx + 1}/{max_rounds} ---")
        improved_any = False

        for name, module in model.named_modules():
            if not isinstance(module, TernaryLinear):
                continue

            group_size = getattr(module, 'group_size', 128)

            with torch.no_grad():
                original_shadow = module.weight.data.clone()

                info = _compute_border_info(
                    module.weight.data, group_size, threshold_coeff
                )

                border_mask = info['distance'] < epsilon

                if info['pad_len'] > 0:
                    flat_mask = border_mask.reshape(-1)
                    flat_mask[info['real_numel']:] = False
                    border_mask = flat_mask.reshape(border_mask.shape)

                ambiguous = info['grouped'].abs() < sign_epsilon
                skip_mask = ambiguous & ~info['is_nonzero']
                border_mask = border_mask & ~skip_mask

                n_border = border_mask.sum().item()
                n_to_zero = (border_mask & info['is_nonzero']).sum().item()
                n_to_nonzero = (border_mask & ~info['is_nonzero']).sum().item()

                if n_border == 0:
                    continue

                flipped = _apply_flips(info, border_mask, push_margin)

                n = module.weight.data.numel()
                new_weight = flipped.reshape(-1)[:n].reshape(original_shadow.shape)
                module.weight.data.copy_(new_weight.to(original_shadow.dtype))

            new_ppl = evaluate_perplexity(
                model, config, split='val', batch_size=batch_size,
                device=device, data_dir=data_dir
            )

            if new_ppl < current_ppl:
                improvement = current_ppl - new_ppl
                current_ppl = new_ppl
                improved_any = True
                total_flips += n_border
                layer_stats.append({
                    'round': round_idx + 1,
                    'layer': name,
                    'n_border': n_border,
                    'n_to_zero': n_to_zero,
                    'n_to_nonzero': n_to_nonzero,
                    'ppl_before': current_ppl + improvement,
                    'ppl_after': current_ppl,
                    'improvement': improvement,
                })
                print(f"  {name}: PPL {current_ppl + improvement:.2f} -> {current_ppl:.2f} "
                      f"(+{improvement:.2f}, {n_border} flipped: {n_to_zero}->0, {n_to_nonzero}->nonzero)")
            else:
                module.weight.data.copy_(original_shadow)
                print(f"  {name}: no improvement ({n_border} border, PPL {new_ppl:.2f})")

        if not improved_any:
            print(f"\nNo improvement in round {round_idx + 1}. Stopping.")
            break

    elapsed = time.time() - start_time
    total_improvement = baseline_ppl - current_ppl
    pct_improvement = 100.0 * total_improvement / baseline_ppl if baseline_ppl > 0 else 0

    print(f"\n{'='*60}")
    print(f"DISCRETE POLISH RESULTS")
    print(f"{'='*60}")
    print(f"Baseline PPL:     {baseline_ppl:.2f}")
    print(f"Final PPL:        {current_ppl:.2f}")
    print(f"Improvement:      {total_improvement:.2f} ({pct_improvement:.2f}%)")
    print(f"Total flips:      {total_flips}")
    print(f"Rounds:           {round_idx + 1}")
    print(f"Time:             {elapsed:.1f}s")
    print(f"{'='*60}")

    return {
        'baseline_ppl': baseline_ppl,
        'final_ppl': current_ppl,
        'improvement': total_improvement,
        'pct_improvement': pct_improvement,
        'total_flips': total_flips,
        'rounds': round_idx + 1,
        'elapsed': elapsed,
        'layer_stats': layer_stats,
        'epsilon': epsilon,
        'push_margin': push_margin,
    }


def polish_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config_name: str = 'wikitext_6m',
    device: str = 'cuda',
    data_dir: str = 'data/wikitext103',
    epsilon: float = 0.1,
    push_margin: float = 0.3,
    max_rounds: int = 10,
    eval_batches: Optional[int] = None,
    batch_size: int = 16,
    use_ternary: bool = True,
    use_ssd: bool = True,
    use_routing: bool = True,
) -> Dict:
    """Load checkpoint, apply discrete polish, save result."""

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

    result = discrete_polish(
        model, config, device=device, data_dir=data_dir,
        epsilon=epsilon, push_margin=push_margin,
        max_rounds=max_rounds, eval_batches=eval_batches,
        batch_size=batch_size,
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
        description='Discrete Polish: post-training ternary weight optimization'
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
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Border distance threshold')
    parser.add_argument('--push-margin', type=float, default=0.3,
                       help='Push margin for 0->nonzero flips')
    parser.add_argument('--max-rounds', type=int, default=10)
    parser.add_argument('--eval-batches', type=int, default=None,
                       help='Limit eval batches (None=full)')
    parser.add_argument('--batch-size', type=int, default=16)
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
        epsilon=args.epsilon,
        push_margin=args.push_margin,
        max_rounds=args.max_rounds,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        use_ternary=not args.no_ternary,
        use_ssd=not args.no_ssd,
        use_routing=not args.no_routing,
    )
