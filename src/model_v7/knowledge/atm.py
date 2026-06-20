from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components.ternary import TernaryLinear, TernaryQuantize, DEFAULT_THRESHOLD_COEFF


class ATMLinear(nn.Module):
    """Adapter with Parallel Ternary Matrix overlay for knowledge injection.

    Wraps a frozen TernaryLinear with a parallel trainable ternary matrix.
    The ATM matrix is added to the base layer output with a scaling factor alpha.
    When alpha=0, ATM computation is skipped for efficiency.
    """

    def __init__(self, base_linear: TernaryLinear, alpha: float = 0.0):
        super().__init__()
        object.__setattr__(self, '_base_linear', base_linear)

        self.atm_weight = nn.Parameter(
            torch.zeros(base_linear.out_features, base_linear.in_features)
        )
        self.alpha = alpha

        if hasattr(base_linear, 'group_size'):
            self.group_size = base_linear.group_size
        else:
            self.group_size = 128

        if hasattr(base_linear, 'threshold_coeff'):
            self.threshold_coeff = base_linear.threshold_coeff
        else:
            self.threshold_coeff = DEFAULT_THRESHOLD_COEFF

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self._base_linear(x)

        if self.alpha == 0.0:
            return base_out

        atm_q = TernaryQuantize.apply(self.atm_weight, self.group_size, self.threshold_coeff)
        atm_out = F.linear(x, atm_q.to(x.device).to(x.dtype))

        return base_out + self.alpha * atm_out


class ATMAdapter:
    """Manager for attaching/detaching ATM adapters to a model.

    Handles lifecycle of ATM layers: attachment to TernaryLinear modules,
    freezing base weights, parameter collection, and detachment.
    """

    def __init__(self):
        self._original_modules: Dict[str, Tuple[nn.Module, str, TernaryLinear]] = {}
        self._atm_modules: Dict[str, ATMLinear] = {}

    def attach(
        self,
        model: nn.Module,
        target_layers: List[str] = None,
        skip_layers: int = 0,
    ) -> List[nn.Parameter]:
        """Attach ATM adapters to TernaryLinear layers in the model.

        Args:
            model: The model to attach adapters to
            target_layers: Optional list of layer name patterns (substring match).
                          If None, attaches to all TernaryLinear layers.
            skip_layers: Number of initial layers to skip (by layer index in name).
                        If > 0, layers with index < skip_layers are not adapted.

        Returns:
            List of all ATM parameters for optimizer
        """
        atm_params = []

        for name, module in model.named_modules():
            if not isinstance(module, TernaryLinear):
                continue

            if target_layers is not None:
                if not any(pattern in name for pattern in target_layers):
                    continue

            if skip_layers > 0:
                layer_index = self._extract_layer_index(name)
                if layer_index is not None and layer_index < skip_layers:
                    continue

            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_path, attr_name = parts
                parent = model.get_submodule(parent_path)
            else:
                parent = model
                attr_name = name

            for param in module.parameters():
                param.requires_grad = False

            atm_linear = ATMLinear(module, alpha=0.0)
            self._original_modules[name] = (parent, attr_name, module)
            self._atm_modules[name] = atm_linear

            setattr(parent, attr_name, atm_linear)
            atm_params.append(atm_linear.atm_weight)

        return atm_params

    @staticmethod
    def _extract_layer_index(name: str) -> int:
        """Extract layer index from module name (e.g., 'layers.5.block.linear' -> 5)."""
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    def detach(self, model: nn.Module) -> None:
        """Restore original TernaryLinear modules and unfreeze their parameters.

        Args:
            model: The model to detach adapters from
        """
        for name, (parent, attr_name, original_module) in self._original_modules.items():
            setattr(parent, attr_name, original_module)
            for param in original_module.parameters():
                param.requires_grad = True

        self._original_modules.clear()
        self._atm_modules.clear()

    def get_atm_params(self) -> List[nn.Parameter]:
        """Get all ATM weight parameters."""
        return [atm.atm_weight for atm in self._atm_modules.values()]

    def set_alpha(self, alpha: float) -> None:
        """Set alpha scaling factor on all ATM modules.

        Args:
            alpha: New alpha value
        """
        for atm in self._atm_modules.values():
            atm.alpha = alpha

    def get_atm_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get all ATM weights as a state dict.

        Returns:
            Dict mapping module name to atm_weight tensor
        """
        return {name: atm.atm_weight.data.clone() for name, atm in self._atm_modules.items()}

    def load_atm_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load ATM weights from a state dict.

        Args:
            state_dict: Dict mapping module name to atm_weight tensor
        """
        for name, weight_data in state_dict.items():
            if name in self._atm_modules:
                self._atm_modules[name].atm_weight.data.copy_(weight_data)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get sparsity statistics for all ATM modules.

        Returns:
            Dict mapping module name to stats dict with:
            - nonzero_count: Number of nonzero weights
            - total_count: Total weights
            - sparsity: Percentage of zero weights
        """
        stats = {}
        for name, atm in self._atm_modules.items():
            with torch.no_grad():
                nonzero = (atm.atm_weight.data != 0).sum().item()
                total = atm.atm_weight.data.numel()
                sparsity = 100.0 * (total - nonzero) / total if total > 0 else 0.0

                stats[name] = {
                    'nonzero_count': nonzero,
                    'total_count': total,
                    'sparsity': sparsity,
                }

        return stats
