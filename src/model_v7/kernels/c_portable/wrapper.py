"""Python wrapper for portable C ternary GEMV kernel."""

import ctypes
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from ...components.ternary import DEFAULT_THRESHOLD_COEFF


_lib = None


def _compile_if_needed():
    kernel_dir = Path(__file__).parent
    from . import compile
    compile.compile()


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib

    kernel_dir = Path(__file__).parent
    output_filename = _get_output_filename()
    lib_path = kernel_dir / output_filename

    if not lib_path.exists():
        print(f"[C-Portable] Compiling kernel...")
        _compile_if_needed()

    if not lib_path.exists():
        raise RuntimeError(f"Failed to compile {output_filename}")

    _lib = ctypes.CDLL(str(lib_path))

    _lib.ternary_gemv_scalar.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.ternary_gemv_scalar.restype = None

    _lib.has_avx2.argtypes = []
    _lib.has_avx2.restype = ctypes.c_int

    _lib.ternary_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.ternary_gemv.restype = None

    return _lib


def _get_output_filename() -> str:
    if sys.platform == 'win32':
        return 'ternary_gemv.dll'
    elif sys.platform == 'darwin':
        return 'ternary_gemv.dylib'
    else:
        return 'ternary_gemv.so'


@dataclass
class PackedCPortable:
    weights_packed: torch.Tensor
    scales: torch.Tensor
    bias: Optional[torch.Tensor]
    in_features: int
    out_features: int
    K_padded: int
    group_size: int
    num_groups: int


def pack_for_c_portable(module, weight_override=None) -> PackedCPortable:
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

    K_packed = (K_padded + 3) // 4
    weights_packed = torch.zeros(N, K_packed, dtype=torch.uint8, device=device)

    for n in range(N):
        for k in range(K_padded):
            byte_idx = k // 4
            bit_offset = (k % 4) * 2
            w_val = ternary_flat[n, k].item()

            if w_val == 1.0:
                w_code = 1
            elif w_val == -1.0:
                w_code = 2
            else:
                w_code = 0

            weights_packed[n, byte_idx] |= (w_code & 0x3) << bit_offset

    bias_tensor = module.bias.data.clone() if module.bias is not None else None

    if hasattr(module, 'alpha') and module.alpha is not None:
        if module.alpha_group_size is None:
            scales = scales * module.alpha.data.item()
        else:
            alpha_flat = module.alpha.data
            if alpha_flat.numel() == num_groups:
                scales = scales * alpha_flat.unsqueeze(0)

    return PackedCPortable(
        weights_packed=weights_packed,
        scales=scales,
        bias=bias_tensor,
        in_features=K,
        out_features=N,
        K_padded=K_padded,
        group_size=group_size,
        num_groups=num_groups,
    )


def c_ternary_gemv(x: torch.Tensor, packed: PackedCPortable) -> torch.Tensor:
    lib = _get_lib()

    if x.device.type != 'cpu':
        x = x.cpu()
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.float()

    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    x_flat = x.reshape(-1, K)
    B = x_flat.shape[0]

    M = packed.out_features
    K_padded = packed.K_padded
    group_size = packed.group_size
    num_groups = packed.num_groups

    output = torch.zeros(B, M, dtype=torch.float32)

    for b in range(B):
        x_b = x_flat[b].contiguous()
        out_b = output[b].contiguous()

        lib.ternary_gemv(
            ctypes.cast(out_b.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(packed.weights_packed.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(x_b.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(packed.scales.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(packed.bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if packed.bias is not None else None,
            M, K_padded, group_size
        )

    return output.reshape(*orig_shape, M)


class CPortableLinear(nn.Module):
    def __init__(self, packed: PackedCPortable, correction_A=None, correction_B=None,
                 block_mixing=None, block_size=0, gate_post=None, gate_pre=None):
        super().__init__()

        self.register_buffer('weights_packed', packed.weights_packed)
        self.register_buffer('scales', packed.scales)

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

        packed = self._get_packed()
        output = c_ternary_gemv(x_flat, packed)

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

    def _get_packed(self) -> PackedCPortable:
        return PackedCPortable(
            weights_packed=self.weights_packed,
            scales=self.scales,
            bias=self.bias_param,
            in_features=self.in_features,
            out_features=self.out_features,
            K_padded=self.K_padded,
            group_size=self.group_size,
            num_groups=self.num_groups,
        )


def convert_model_to_c_portable(model, verbose=True) -> Tuple[nn.Module, int, int]:
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

            packed = pack_for_c_portable(module, weight_override=weight_override)

            correction_A = getattr(module, 'correction_A', None)
            correction_B = getattr(module, 'correction_B', None)
            block_mixing = getattr(module, 'block_mixing', None)
            block_size = getattr(module, 'block_size', 0)
            gate_post = getattr(module, 'gate_post', None)
            gate_pre = getattr(module, 'gate_pre', None)

            replacement = CPortableLinear(
                packed,
                correction_A=correction_A,
                correction_B=correction_B,
                block_mixing=block_mixing,
                block_size=block_size,
                gate_post=gate_post,
                gate_pre=gate_pre
            )

            if verbose and (gate_post is not None or correction_A is not None or
                           block_mixing is not None or gate_pre is not None):
                print(f"[C-Portable] Layer {name}: extra features "
                      f"(gate_post={gate_post is not None}, gate_pre={gate_pre is not None}, "
                      f"correction={correction_A is not None}, block_mixing={block_mixing is not None})")

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            n_converted += 1

    if verbose:
        print(f"[C-Portable] Converted {n_converted} layers, skipped {n_skipped} (quantize_enabled=False)")

    return model, n_converted, n_skipped
