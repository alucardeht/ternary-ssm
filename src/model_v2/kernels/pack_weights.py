from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PackedLayer:
    w_packed: torch.Tensor
    scales: torch.Tensor
    bias: Optional[torch.Tensor]
    in_features: int
    out_features: int
    group_size: int


def pack_ternary_linear(module, group_size=128):
    weight = module.weight.data
    N, K = weight.shape
    device = weight.device  # Preserve device

    pad_to = max(group_size, 16)
    K_padded = ((K + pad_to - 1) // pad_to) * pad_to
    if K_padded > K:
        weight = F.pad(weight, (0, K_padded - K))

    grouped = weight.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    threshold = 0.7 * normalized.abs().mean(dim=-1, keepdim=True)
    ternary = normalized.sign() * (normalized.abs() > threshold).float()

    num_groups = K_padded // group_size
    scales = scales.squeeze(-1).reshape(N, num_groups)

    ternary_flat = ternary.reshape(N, K_padded)
    encoded = torch.zeros_like(ternary_flat, dtype=torch.int32, device=device)
    encoded[ternary_flat == 1] = 1
    encoded[ternary_flat == -1] = 2

    K_packed = K_padded // 16
    encoded_reshaped = encoded.reshape(N, K_packed, 16)
    packed = torch.zeros(N, K_packed, dtype=torch.int32, device=device)
    for i in range(16):
        packed |= (encoded_reshaped[:, :, i] & 0x3) << (i * 2)

    bias_tensor = module.bias.data.clone() if module.bias is not None else None

    return PackedLayer(
        w_packed=packed,
        scales=scales,
        bias=bias_tensor,
        in_features=K,
        out_features=N,
        group_size=group_size,
    )


def unpack_ternary_weights(packed_layer):
    N = packed_layer.out_features
    K_packed = packed_layer.w_packed.shape[1]
    K_padded = K_packed * 16

    unpacked = torch.zeros(N, K_packed, 16, dtype=torch.float32)
    for i in range(16):
        bits = (packed_layer.w_packed >> (i * 2)) & 0x3
        vals = torch.zeros_like(bits, dtype=torch.float32)
        vals[bits == 1] = 1.0
        vals[bits == 2] = -1.0
        unpacked[:, :, i] = vals

    return unpacked.reshape(N, K_padded)[:, :packed_layer.in_features]


def quantize_activations_int8(x):
    abs_max = x.abs().max().clamp(min=1e-8)
    act_scale = abs_max / 127.0
    x_int8 = (x / act_scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, act_scale.item()


def pack_model(model):
    from ..components.ternary import TernaryLinear
    packed = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            packed[name] = pack_ternary_linear(module, module.group_size)
    return packed


class TernaryLinearPacked(nn.Module):

    def __init__(self, packed_layer, kernel):
        super().__init__()
        self.packed_layer = packed_layer
        self.kernel = kernel
        if packed_layer.bias is not None:
            self.register_buffer('bias', packed_layer.bias)
        self.register_buffer('w_packed', packed_layer.w_packed)
        self.register_buffer('scales', packed_layer.scales)

    def forward(self, x):
        return self.kernel.matmul(x, self.packed_layer)


def convert_model_for_cuda_inference(model, kernel):
    from ..components.ternary import TernaryLinear
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            packed = pack_ternary_linear(module, module.group_size)
            replacement = TernaryLinearPacked(packed, kernel)
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
    return model
