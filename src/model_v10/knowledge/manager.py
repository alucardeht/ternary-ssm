import torch
import json
from pathlib import Path
from typing import Dict, Optional, List

from .atm import ATMAdapter
from ..components.ternary import DEFAULT_THRESHOLD_COEFF


class ATMManager:
    """Manages persistence and lifecycle of ATM (Additive Ternary Module) adapters."""

    @staticmethod
    def save(adapter: ATMAdapter, path: str, metadata: Optional[Dict] = None) -> None:
        """Save ATM weights and metadata to file.

        Args:
            adapter: ATMAdapter instance with trained modules
            path: File path to save to
            metadata: Optional metadata dict to save with weights
        """
        state = {
            'atm_state': adapter.get_atm_state_dict(),
            'metadata': metadata or {},
            'version': 1,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        size_kb = Path(path).stat().st_size / 1024
        print(f"ATM saved to {path} ({size_kb:.1f} KB)")

    @staticmethod
    def load(model, path: str, target_layers: Optional[List[str]] = None) -> ATMAdapter:
        """Load ATM weights from file and attach to model.

        Args:
            model: Model to attach loaded ATM to
            path: File path to load from
            target_layers: Optional layer name patterns

        Returns:
            ATMAdapter with loaded weights
        """
        state = torch.load(path, map_location='cpu', weights_only=False)
        adapter = ATMAdapter()
        adapter.attach(model, target_layers)
        adapter.load_atm_state_dict(state['atm_state'])
        adapter.set_alpha(1.0)
        print(f"ATM loaded from {path} (version {state.get('version', 'unknown')})")
        return adapter

    @staticmethod
    def merge(model, adapter: ATMAdapter) -> None:
        """Permanently merge ATM knowledge into base model weights.

        Args:
            model: Model containing base TernaryLinear modules
            adapter: ATMAdapter with trained weights to merge
        """
        from ..components.ternary import TernaryQuantize

        merged_count = 0
        for name, atm_module in adapter._atm_modules.items():
            base = atm_module._base_linear

            atm_q = TernaryQuantize.apply(
                atm_module.atm_weight,
                getattr(atm_module, 'group_size', 64),
                getattr(atm_module, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF)
            ).detach()

            base.weight.data.add_(atm_q.to(base.weight.dtype))
            merged_count += 1

        adapter.detach(model)
        print(f"Merged {merged_count} ATM modules into base model")

    @staticmethod
    def remove(model, adapter: ATMAdapter) -> None:
        """Remove ATM adapter without merging (discard knowledge).

        Args:
            model: Model to remove adapters from
            adapter: ATMAdapter to remove
        """
        adapter.detach(model)
        print("ATM adapter removed (knowledge discarded)")

    @staticmethod
    def list_modules(directory: str) -> List[Dict]:
        """List all saved ATM modules in a directory.

        Args:
            directory: Directory to search for saved ATM files

        Returns:
            List of dicts with module metadata
        """
        results = []
        for path in Path(directory).glob('*.pt'):
            try:
                state = torch.load(path, map_location='cpu', weights_only=False)
                meta = state.get('metadata', {})
                meta['path'] = str(path)
                meta['size_kb'] = path.stat().st_size / 1024
                results.append(meta)
            except Exception:
                continue
        return results
