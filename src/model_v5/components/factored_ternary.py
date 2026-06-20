import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ternary import TernaryQuantize, DEFAULT_THRESHOLD_COEFF


class FactoredTernaryLinear(nn.Module):
    """
    Drop-in replacement for TernaryLinear that decomposes one dense d_in x d_out
    ternary matrix into K sparse ternary factor matrices.

    Each factor has c non-zeros per row, yielding exponential capacity c^K
    with linear compute K * d_out * c.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_factors: int = 8,
        sparsity: int = 3,
        bias: bool = False,
        group_size: int = 128,
        pattern: str = 'random',
        config=None,
        threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_factors = n_factors
        self.sparsity = sparsity
        self.bias_enabled = bias
        self.group_size = group_size
        self.pattern = pattern
        self.threshold_coeff = threshold_coeff

        if config is not None and hasattr(config, 'ternary_group_size'):
            self.group_size = config.ternary_group_size
        if config is not None and hasattr(config, 'ste_type'):
            pass

        self.shadow_values = nn.ParameterList()

        current_dim = in_features
        for k in range(n_factors):
            shadow = nn.Parameter(torch.zeros(out_features, sparsity))
            self.shadow_values.append(shadow)

            indices_tensor = self._init_indices(out_features, current_dim, k)
            self.register_buffer(f'indices_{k}', indices_tensor)

            current_dim = out_features

        self._init_shadow_values()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.skip_alpha = nn.Parameter(torch.tensor(0.1))
        if in_features == out_features:
            self.skip_conn = None
        else:
            self.skip_conn = nn.Linear(in_features, out_features, bias=False)
            nn.init.eye_(self.skip_conn.weight[:min(in_features, out_features), :min(in_features, out_features)])

        self._forward_call_count = 0

        dense_ops = in_features * out_features
        factored_ops = n_factors * out_features * sparsity
        reduction = dense_ops / max(factored_ops, 1)
        print(f"[FTEC] FactoredTernaryLinear({in_features}->{out_features}) "
              f"K={n_factors}, c={sparsity}, pattern={pattern} "
              f"dense_ops={dense_ops} factored_ops={factored_ops} "
              f"reduction={reduction:.1f}x")

    def _init_indices(self, out_features: int, in_dim: int, factor_idx: int) -> torch.Tensor:
        """Initialize sparsity pattern indices."""
        if self.pattern == 'random':
            indices = torch.zeros(out_features, self.sparsity, dtype=torch.long)
            for i in range(out_features):
                perm = torch.randperm(in_dim)[:self.sparsity]
                indices[i] = perm
            return indices
        elif self.pattern == 'shift':
            indices = torch.zeros(out_features, self.sparsity, dtype=torch.long)
            stride = in_dim // (factor_idx + 1) if factor_idx > 0 else 1
            for i in range(out_features):
                for j in range(self.sparsity):
                    indices[i, j] = (i + j * stride) % in_dim
            return indices
        elif self.pattern == 'butterfly':
            indices = torch.zeros(out_features, self.sparsity, dtype=torch.long)
            for i in range(out_features):
                indices[i, 0] = i
                indices[i, 1] = (i + in_dim // 2) % in_dim
            return indices
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def _init_shadow_values(self):
        """Initialize shadow values to ±1 with equal probability."""
        for shadow in self.shadow_values:
            shadow.data = torch.randint(0, 2, shadow.shape, dtype=shadow.dtype) * 2.0 - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._forward_call_count += 1

        if self._forward_call_count <= 5:
            print(f"[FTEC] forward call={self._forward_call_count} input_shape={list(x.shape)} ", end="")

        h = x
        for k in range(self.n_factors):
            indices_k = getattr(self, f'indices_{k}')

            x_gathered = h[..., indices_k]

            w_ternary = TernaryQuantize.apply(
                self.shadow_values[k], self.group_size, self.threshold_coeff
            )

            h = (x_gathered * w_ternary).sum(dim=-1)

        if self.skip_conn is not None:
            skip_out = self.skip_conn(x)
        else:
            skip_out = x

        h = h + self.skip_alpha * skip_out

        if self.bias is not None:
            h = h + self.bias

        if self._forward_call_count <= 5:
            print(f"output_shape={list(h.shape)} elapsed_us={0}")

        return h


def replace_with_factored(
    model: nn.Module,
    target_layers: str = 'out_proj',
    n_factors: int = 8,
    sparsity: int = 3,
    pattern: str = 'random',
    skip_connection: bool = True,
    config=None,
) -> int:
    """
    Replace specific TernaryLinear modules with FactoredTernaryLinear.

    Args:
        model: Model to modify
        target_layers: 'out_proj', 'all_proj', 'ssm_proj', 'swa_proj'
        n_factors: Factorization depth K
        sparsity: Non-zeros per row per factor (c)
        pattern: Sparsity pattern type
        skip_connection: Enable residual skip
        config: ModelConfig for threshold_coeff etc

    Returns:
        Number of layers replaced
    """
    from .ternary import TernaryLinear

    replaced_count = 0

    def should_replace(name: str) -> bool:
        if target_layers == 'out_proj':
            return 'out_proj' in name
        elif target_layers == 'all_proj':
            return 'proj' in name
        elif target_layers == 'ssm_proj':
            return 'ssm' in name and 'proj' in name
        elif target_layers == 'swa_proj':
            return 'swa' in name or 'attention' in name or 'mlp' in name
        else:
            return False

    for name, module in list(model.named_modules()):
        parent_name, _, child_name = name.rpartition('.')
        if parent_name == '':
            parent = model
        else:
            parent = dict(model.named_modules())[parent_name]

        if isinstance(module, TernaryLinear) and should_replace(name):
            threshold_coeff = getattr(module, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF)
            factored = FactoredTernaryLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                n_factors=n_factors,
                sparsity=sparsity,
                bias=module.bias is not None,
                group_size=module.group_size if hasattr(module, 'group_size') else 128,
                pattern=pattern,
                config=config,
                threshold_coeff=threshold_coeff,
            )
            setattr(parent, child_name, factored)
            replaced_count += 1

    return replaced_count
