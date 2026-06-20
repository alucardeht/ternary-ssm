import argparse
import re
import torch
from dataclasses import replace
from pathlib import Path
from typing import Dict

from src.model_v11.config import (
    ModelConfig,
    config_1m,
    config_10m,
    config_100m,
    config_scale_10m,
    config_scale_50m,
    config_scale_100m,
    config_wikitext_3m,
    config_wikitext_6m,
    config_wikitext_50m,
    config_wikitext_100m,
    config_wikitext_bench,
)
from src.model_v11.model import ModelV2


CONFIG_REGISTRY = {
    '1m': config_1m,
    '10m': config_10m,
    '100m': config_100m,
    'scale_10m': config_scale_10m,
    'scale_50m': config_scale_50m,
    'scale_100m': config_scale_100m,
    'wikitext_3m': config_wikitext_3m,
    'wikitext_6m': config_wikitext_6m,
    'wikitext_50m': config_wikitext_50m,
    'wikitext_100m': config_wikitext_100m,
    'wikitext_bench': config_wikitext_bench,
}


def _should_add_noise(key: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim < 2:
        return False
    if 'weight' not in key:
        return False
    exclude_keywords = ('norm', 'conv1d', 'dt_proj', 'rotary', 'router', 'embedding')
    if any(kw in key for kw in exclude_keywords):
        return False
    return True


def _load_checkpoint(path: str) -> dict:
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


def grow_model(
    checkpoint_path: str,
    output_path: str,
    g: int = 2,
    noise_std: float = 0.01,
    config_name: str = 'wikitext_6m',
    use_ternary: bool = True,
    use_ssd: bool = True,
    use_routing: bool = True,
) -> Dict:
    print(f"\nLoading checkpoint from {checkpoint_path}")
    state_dict = _load_checkpoint(checkpoint_path)

    print(f"Getting config: {config_name}")
    config = CONFIG_REGISTRY[config_name]()

    config.use_ternary = use_ternary
    config.use_ssd = use_ssd
    config.use_routing = use_routing
    config.use_torch_compile = False

    print(f"Original config: n_layer={config.n_layer}, swa_indices={config.swa_layer_indices}")
    original_params = sum(
        p.numel() for p in ModelV2(config).parameters()
    )
    print(f"Original params: {original_params:,}")

    new_swa_indices = []
    for idx in config.swa_layer_indices:
        new_swa_indices.extend([idx * g + j for j in range(g)])

    grown_config = replace(
        config,
        n_layer=config.n_layer * g,
        swa_layer_indices=new_swa_indices,
        use_ternary=use_ternary,
        use_ssd=use_ssd,
        use_routing=use_routing,
        use_torch_compile=False,
    )

    print(f"Grown config: n_layer={grown_config.n_layer}, swa_indices={grown_config.swa_layer_indices}")

    grown_state_dict = {}
    layer_pattern = re.compile(r'^layers\.(\d+)\.(.*)')
    noise_injected_params = 0

    for key, tensor in state_dict.items():
        match = layer_pattern.match(key)

        if not match:
            grown_state_dict[key] = tensor.clone()
            continue

        old_layer_idx = int(match.group(1))
        suffix = match.group(2)
        base_position = old_layer_idx * g

        grown_state_dict[f'layers.{base_position}.{suffix}'] = tensor.clone()

        for offset in range(1, g):
            new_layer_idx = base_position + offset
            new_key = f'layers.{new_layer_idx}.{suffix}'

            cloned_tensor = tensor.clone()

            if _should_add_noise(key, tensor):
                noise = torch.randn_like(cloned_tensor) * noise_std
                cloned_tensor = cloned_tensor + noise
                noise_injected_params += cloned_tensor.numel()

            grown_state_dict[new_key] = cloned_tensor

    print(f"\nBuilding grown model...")
    grown_model = ModelV2(grown_config)

    print(f"Loading grown state dict...")
    missing, unexpected = grown_model.load_state_dict(grown_state_dict, strict=True)

    grown_params = sum(p.numel() for p in grown_model.parameters())
    print(f"\nGrown model params: {grown_params:,}")
    print(f"Noise-injected params: {noise_injected_params:,}")
    print(f"Growth factor: {grown_params / original_params:.2f}x")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'model_state': grown_model.state_dict(),
        'config_name': config_name,
        'growth_metadata': {
            'original_n_layer': config.n_layer,
            'grown_n_layer': grown_config.n_layer,
            'growth_factor': g,
            'original_params': original_params,
            'grown_params': grown_params,
            'noise_std': noise_std,
            'noise_injected_params': noise_injected_params,
            'original_swa_indices': config.swa_layer_indices,
            'grown_swa_indices': grown_config.swa_layer_indices,
        },
    }

    torch.save(metadata, output_path)
    print(f"\nSaved grown model to {output_path}")

    return metadata['growth_metadata']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grow model by duplicating layers')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path for grown model')
    parser.add_argument('--g', type=int, default=2, help='Growth factor (default: 2)')
    parser.add_argument('--noise-std', type=float, default=0.01, help='Noise std for duplication')
    parser.add_argument(
        '--config',
        type=str,
        default='wikitext_6m',
        choices=list(CONFIG_REGISTRY.keys()),
        help='Config name',
    )
    parser.add_argument(
        '--no-ternary', action='store_true', help='Disable ternary quantization'
    )
    parser.add_argument(
        '--no-ssd', action='store_true', help='Disable SSD'
    )
    parser.add_argument(
        '--no-routing', action='store_true', help='Disable MoD routing'
    )

    args = parser.parse_args()

    grow_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        g=args.g,
        noise_std=args.noise_std,
        config_name=args.config,
        use_ternary=not args.no_ternary,
        use_ssd=not args.no_ssd,
        use_routing=not args.no_routing,
    )
