"""T-SAR: Ternary Subtract-Add-Reduce CPU inference kernel.

Two matmul strategies:
- Dense: Pre-baked float32 ternary matrix, uses F.linear (MKL-optimized)
- Scatter: Index-gather + scatter_add, truly multiplication-free

Auto-selects scatter for B=1 (autoregressive) and dense for B>1 (batch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from ..components.ternary import DEFAULT_THRESHOLD_COEFF


@dataclass
class PackedTernaryCPU:
    w_val: torch.Tensor
    w_sign: torch.Tensor
    scales: torch.Tensor
    bias: Optional[torch.Tensor]
    w_dense: torch.Tensor
    pos_cols: torch.Tensor
    pos_targets: torch.Tensor
    neg_cols: torch.Tensor
    neg_targets: torch.Tensor
    in_features: int
    out_features: int
    K_padded: int
    group_size: int
    num_groups: int


def pack_ternary_cpu(module, weight_override=None) -> PackedTernaryCPU:
    weight = weight_override if weight_override is not None else module.weight.data
    N, K = weight.shape
    device = weight.device
    group_size = module.group_size

    pad_to = max(group_size, 16)
    K_padded = ((K + pad_to - 1) // pad_to) * pad_to
    if K_padded > K:
        weight = F.pad(weight, (0, K_padded - K))

    grouped = weight.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    threshold = DEFAULT_THRESHOLD_COEFF * normalized.abs().mean(dim=-1, keepdim=True)
    ternary = normalized.sign() * (normalized.abs() > threshold).float()

    num_groups = K_padded // group_size
    scales = scales.squeeze(-1).reshape(N, num_groups)

    ternary_flat = ternary.reshape(N, K_padded)

    K_packed = K_padded // 32
    ternary_reshaped = ternary_flat.reshape(N, K_packed, 32)

    nonzero = (ternary_reshaped != 0).int()
    negative = (ternary_reshaped < 0).int()

    bit_positions = (1 << torch.arange(32, device=device, dtype=torch.int32))
    w_val = (nonzero * bit_positions).sum(dim=-1).int()
    w_sign = (negative * bit_positions).sum(dim=-1).int()

    bias_tensor = module.bias.data.clone() if module.bias is not None else None

    if hasattr(module, 'alpha') and module.alpha is not None:
        if module.alpha_group_size is None:
            scales = scales * module.alpha.data.item()
        else:
            alpha_flat = module.alpha.data
            num_alpha_groups = (K_padded + module.alpha_group_size - 1) // module.alpha_group_size
            if alpha_flat.numel() == num_groups:
                scales = scales * alpha_flat.unsqueeze(0)

    w_dense = _unpack_and_scale_ternary(w_val, w_sign, scales, N, K_padded, group_size, device)

    pos_cols, pos_targets, neg_cols, neg_targets = _build_scatter_indices(
        ternary_flat, N, group_size
    )

    return PackedTernaryCPU(
        w_val=w_val,
        w_sign=w_sign,
        scales=scales,
        bias=bias_tensor,
        w_dense=w_dense,
        pos_cols=pos_cols,
        pos_targets=pos_targets,
        neg_cols=neg_cols,
        neg_targets=neg_targets,
        in_features=K,
        out_features=N,
        K_padded=K_padded,
        group_size=group_size,
        num_groups=num_groups,
    )


def _unpack_and_scale_ternary(w_val, w_sign, scales, N, K_padded, group_size, device):
    bit_offsets = torch.arange(32, device=device, dtype=torch.int32)
    val_expanded = ((w_val.unsqueeze(-1) >> bit_offsets.view(1, 1, 32)) & 1).float()
    sign_expanded = ((w_sign.unsqueeze(-1) >> bit_offsets.view(1, 1, 32)) & 1).float()

    ternary = val_expanded * (1.0 - 2.0 * sign_expanded)
    ternary = ternary.reshape(N, K_padded)

    num_groups = K_padded // group_size
    scales_expanded = scales.unsqueeze(-1).expand(-1, -1, group_size)
    scales_flat = scales_expanded.reshape(N, K_padded)
    w_dense = ternary * scales_flat

    return w_dense.to(torch.float32)


def _build_scatter_indices(ternary_flat, N, group_size):
    pos_mask = (ternary_flat == 1.0)
    neg_mask = (ternary_flat == -1.0)

    pos_rows, pos_cols_tmp = torch.where(pos_mask)
    neg_rows, neg_cols_tmp = torch.where(neg_mask)

    pos_cols = pos_cols_tmp.to(torch.int64)
    neg_cols = neg_cols_tmp.to(torch.int64)

    pos_targets = (pos_rows * (ternary_flat.shape[-1] // group_size) + (pos_cols_tmp // group_size)).to(torch.int64)
    neg_targets = (neg_rows * (ternary_flat.shape[-1] // group_size) + (neg_cols_tmp // group_size)).to(torch.int64)

    return pos_cols, pos_targets, neg_cols, neg_targets


def ternary_matmul_dense(x: torch.Tensor, packed: PackedTernaryCPU) -> torch.Tensor:
    K_padded = packed.K_padded
    K = packed.in_features

    x_flat = x.reshape(-1, K)
    if x_flat.shape[-1] < K_padded:
        x_flat = F.pad(x_flat, (0, K_padded - x_flat.shape[-1]))

    output = F.linear(x_flat, packed.w_dense, packed.bias)
    return output


def ternary_matmul_scatter(x: torch.Tensor, packed: PackedTernaryCPU) -> torch.Tensor:
    K = packed.in_features
    N = packed.out_features
    group_size = packed.group_size
    num_groups = packed.num_groups

    x_flat = x.reshape(-1)
    if x_flat.shape[0] < packed.K_padded:
        x_flat = F.pad(x_flat, (0, packed.K_padded - x_flat.shape[0]))

    device = packed.w_val.device
    y_grouped = torch.zeros(N * num_groups, dtype=torch.float32, device=device)

    if packed.pos_cols.numel() > 0:
        pos_vals = x_flat[packed.pos_cols].to(torch.float32)
        y_grouped.scatter_add_(0, packed.pos_targets, pos_vals)

    if packed.neg_cols.numel() > 0:
        neg_vals = x_flat[packed.neg_cols].to(torch.float32)
        y_grouped.scatter_add_(0, packed.neg_targets, -neg_vals)

    y = (y_grouped.reshape(N, num_groups) * packed.scales).sum(dim=-1)

    if packed.bias is not None:
        y = y + packed.bias

    return y


class TernaryCPULinear(nn.Module):
    def __init__(self, packed: PackedTernaryCPU, correction_A=None, correction_B=None,
                 block_mixing=None, block_size=0, gate_post=None, gate_pre=None):
        super().__init__()
        self.register_buffer('w_val', packed.w_val)
        self.register_buffer('w_sign', packed.w_sign)
        self.register_buffer('scales', packed.scales)
        self.register_buffer('w_dense', packed.w_dense)
        self.register_buffer('pos_cols', packed.pos_cols)
        self.register_buffer('pos_targets', packed.pos_targets)
        self.register_buffer('neg_cols', packed.neg_cols)
        self.register_buffer('neg_targets', packed.neg_targets)

        if packed.bias is not None:
            self.register_buffer('bias_param', packed.bias)
        else:
            self.bias_param = None

        self.in_features = packed.in_features
        self.out_features = packed.out_features
        self.K_padded = packed.K_padded
        self.group_size = packed.group_size
        self.num_groups = packed.num_groups
        self.block_size = block_size
        self.force_mode = None

        if correction_A is not None:
            self.correction_A = nn.Parameter(correction_A.clone().detach())
        else:
            self.correction_A = None

        if correction_B is not None:
            self.correction_B = nn.Parameter(correction_B.clone().detach())
        else:
            self.correction_B = None

        if block_mixing is not None:
            self.block_mixing = nn.Parameter(block_mixing.clone().detach())
        else:
            self.block_mixing = None

        if gate_post is not None:
            self.gate_post = nn.Parameter(gate_post.clone().detach())
        else:
            self.gate_post = None

        if gate_pre is not None:
            self.gate_pre = nn.Parameter(gate_pre.clone().detach())
        else:
            self.gate_pre = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        orig_shape = x.shape[:-1]
        K = x.shape[-1]
        x_flat = x.reshape(-1, K)
        B = x_flat.shape[0]

        if self.force_mode == 'scatter' or (self.force_mode is None and B == 1):
            output = ternary_matmul_scatter(x_flat[0], self._get_packed())
            output = output.unsqueeze(0)
        else:
            output = ternary_matmul_dense(x_flat, self._get_packed())

        if self.block_mixing is not None:
            output = output.reshape(*orig_shape, self.out_features)
            output = torch.einsum('...gj,gij->...gi', output, self.block_mixing)
            output = output.reshape(-1, self.out_features)

        if self.gate_post is not None:
            output = output * self.gate_post

        if self.correction_A is not None:
            x_orig = x_input.reshape(-1, K)
            correction_term = F.linear(F.linear(x_orig, self.correction_B), self.correction_A)
            output = output.reshape(-1, self.out_features) + correction_term

        return output.reshape(*orig_shape, self.out_features)

    def _get_packed(self) -> PackedTernaryCPU:
        return PackedTernaryCPU(
            w_val=self.w_val,
            w_sign=self.w_sign,
            scales=self.scales,
            bias=self.bias_param,
            w_dense=self.w_dense,
            pos_cols=self.pos_cols,
            pos_targets=self.pos_targets,
            neg_cols=self.neg_cols,
            neg_targets=self.neg_targets,
            in_features=self.in_features,
            out_features=self.out_features,
            K_padded=self.K_padded,
            group_size=self.group_size,
            num_groups=self.num_groups,
        )


def convert_model_to_tsar_cpu(model, verbose=True) -> Tuple[nn.Module, int, int]:
    n_converted = 0
    n_skipped = 0

    for name, module in list(model.named_modules()):
        is_ternary = (hasattr(module, 'quantize_enabled') and
                      hasattr(module, 'group_size') and
                      hasattr(module, 'ste_type'))

        if is_ternary:
            if not getattr(module, 'quantize_enabled', True):
                n_skipped += 1
                continue

            weight_override = None
            if hasattr(module, 'gate_pre') and module.gate_pre is not None:
                weight_override = module.weight.data * module.gate_pre.unsqueeze(0)

            packed = pack_ternary_cpu(module, weight_override=weight_override)

            correction_A = getattr(module, 'correction_A', None)
            correction_B = getattr(module, 'correction_B', None)
            block_mixing = getattr(module, 'block_mixing', None)
            block_size = getattr(module, 'block_size', 0)
            gate_post = getattr(module, 'gate_post', None)
            gate_pre = getattr(module, 'gate_pre', None)

            replacement = TernaryCPULinear(
                packed,
                correction_A=correction_A,
                correction_B=correction_B,
                block_mixing=block_mixing,
                block_size=block_size,
                gate_post=gate_post,
                gate_pre=gate_pre
            )

            if verbose and (gate_post is not None or correction_A is not None or block_mixing is not None):
                print(f"[T-SAR] Layer {name}: extra features (gate_post={gate_post is not None}, "
                      f"correction={correction_A is not None}, block_mixing={block_mixing is not None})")

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            n_converted += 1

    if verbose:
        print(f"[T-SAR] Converted {n_converted} layers, skipped {n_skipped} (quantize_enabled=False)")

    return model, n_converted, n_skipped
