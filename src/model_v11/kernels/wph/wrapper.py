"""Python wrapper for WPH (Weight Pattern Hashing) ternary GEMV kernel."""

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
        print(f"[WPH] Compiling kernel...")
        _compile_if_needed()

    if not lib_path.exists():
        raise RuntimeError(f"Failed to compile {output_filename}")

    _lib = ctypes.CDLL(str(lib_path))

    _lib.wph_encode.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int, ctypes.c_int
    ]
    _lib.wph_encode.restype = None

    _lib.wph_gemv_scalar.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.wph_gemv_scalar.restype = None

    _lib.wph_has_avx2.argtypes = []
    _lib.wph_has_avx2.restype = ctypes.c_int

    _lib.wph_gemv.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _lib.wph_gemv.restype = None

    return _lib


def _get_output_filename() -> str:
    if sys.platform == 'win32':
        return 'wph_gemv.dll'
    elif sys.platform == 'darwin':
        return 'wph_gemv.dylib'
    else:
        return 'wph_gemv.so'


def _wph_encode_python(ternary_int8: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Python fallback for wph_encode if C library unavailable."""
    num_blocks = (M + 4 - 1) // 4
    pattern_ids = torch.zeros((K, num_blocks), dtype=torch.uint8, device=ternary_int8.device)

    for col in range(K):
        for b in range(num_blocks):
            pattern = 0
            power = 1
            rows_in_block = min(4, M - b * 4)

            for r in range(rows_in_block):
                row = b * 4 + r
                val = ternary_int8[row, col].item()
                digit = int(val) + 1
                pattern += digit * power
                power *= 3

            pattern_ids[col, b] = pattern

    return pattern_ids


@dataclass
class PackedWPH:
    pattern_ids: torch.Tensor
    scales: torch.Tensor
    bias: Optional[torch.Tensor]
    in_features: int
    out_features: int
    K_padded: int
    M_padded: int
    group_size: int
    num_groups: int
    num_blocks: int


def pack_for_wph(module, weight_override=None) -> PackedWPH:
    weight = weight_override if weight_override is not None else module.weight.data
    M, K = weight.shape
    device = weight.device
    group_size = module.group_size if hasattr(module, 'group_size') else 128
    threshold_coeff = module.threshold_coeff if hasattr(module, 'threshold_coeff') else DEFAULT_THRESHOLD_COEFF

    pad_to = max(group_size, 16)
    K_padded = ((K + pad_to - 1) // pad_to) * pad_to

    if K_padded > K:
        weight = F.pad(weight, (0, K_padded - K))

    # === Replicate ternary quantization to get signs and scales separately ===
    # This avoids the round-trip precision loss of ternary_quantize_ste which
    # returns (signs * scales) combined, requiring unreliable re-decomposition.
    with torch.no_grad():
        grouped = weight.reshape(-1, group_size)
        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales
        threshold = threshold_coeff * normalized.abs().mean(dim=-1, keepdim=True)
        signs = normalized.sign() * (normalized.abs() > threshold).float()
        ternary_int8 = signs.reshape(M, K_padded).round().to(torch.int8)
        num_groups = K_padded // group_size
        scales_2d = scales.squeeze(-1).reshape(M, num_groups)

    # Apply alpha to scales if present
    if hasattr(module, 'alpha') and module.alpha is not None:
        alpha = module.alpha.data
        if alpha.dim() == 0:
            scales_2d = scales_2d * alpha
        else:
            scales_2d = scales_2d * alpha.unsqueeze(-1)

    scales = scales_2d

    M_padded = ((M + 4 - 1) // 4) * 4
    if M_padded > M:
        ternary_int8 = F.pad(ternary_int8, (0, 0, 0, M_padded - M))
        scales = F.pad(scales, (0, 0, 0, M_padded - M))

    num_blocks = (M_padded + 4 - 1) // 4

    try:
        lib = _get_lib()
        pattern_ids = torch.zeros(num_blocks * K_padded, dtype=torch.uint8, device='cpu')
        lib.wph_encode(
            ctypes.cast(pattern_ids.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(ternary_int8.cpu().data_ptr(), ctypes.POINTER(ctypes.c_int8)),
            M_padded, K_padded
        )
    except Exception as e:
        print(f"[WPH] C library encoding failed ({e}), using Python fallback")
        pattern_ids = _wph_encode_python(ternary_int8, M_padded, K_padded)

    bias_tensor = module.bias.data.clone() if module.bias is not None else None

    return PackedWPH(
        pattern_ids=pattern_ids.cpu().contiguous(),
        scales=scales.cpu().contiguous().float(),
        bias=bias_tensor.cpu().contiguous().float() if bias_tensor is not None else None,
        in_features=K,
        out_features=M,
        K_padded=K_padded,
        M_padded=M_padded,
        group_size=group_size,
        num_groups=num_groups,
        num_blocks=num_blocks,
    )


def wph_ternary_gemv(x: torch.Tensor, packed: PackedWPH) -> torch.Tensor:
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
    M_padded = packed.M_padded
    K_padded = packed.K_padded
    group_size = packed.group_size
    num_groups = packed.num_groups

    output = torch.zeros(B, M_padded, dtype=torch.float32)

    for b in range(B):
        x_b = x_flat[b].contiguous()
        out_b = output[b].contiguous()

        lib.wph_gemv(
            ctypes.cast(out_b.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(packed.pattern_ids.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            ctypes.cast(x_b.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(packed.scales.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(packed.bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if packed.bias is not None else None,
            M_padded, K_padded, group_size
        )

    return output[:, :M].reshape(*orig_shape, M)


class WPHLinear(nn.Module):
    def __init__(self, packed: PackedWPH, correction_A=None, correction_B=None,
                 block_mixing=None, block_size=0, gate_post=None, gate_pre=None):
        super().__init__()

        self.register_buffer('pattern_ids', packed.pattern_ids)
        self.register_buffer('scales', packed.scales)

        if packed.bias is not None:
            self.register_buffer('bias_param', packed.bias)
        else:
            self.bias_param = None

        self.in_features = packed.in_features
        self.out_features = packed.out_features
        self.K_padded = packed.K_padded
        self.M_padded = packed.M_padded
        self.group_size = packed.group_size
        self.num_groups = packed.num_groups
        self.num_blocks = packed.num_blocks
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
        output = wph_ternary_gemv(x_flat, packed)

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

    def _get_packed(self) -> PackedWPH:
        return PackedWPH(
            pattern_ids=self.pattern_ids,
            scales=self.scales,
            bias=self.bias_param,
            in_features=self.in_features,
            out_features=self.out_features,
            K_padded=self.K_padded,
            M_padded=self.M_padded,
            group_size=self.group_size,
            num_groups=self.num_groups,
            num_blocks=self.num_blocks,
        )


def convert_model_to_wph(model, verbose=True) -> Tuple[nn.Module, int, int]:
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

            packed = pack_for_wph(module, weight_override=weight_override)

            correction_A = getattr(module, 'correction_A', None)
            correction_B = getattr(module, 'correction_B', None)
            block_mixing = getattr(module, 'block_mixing', None)
            block_size = getattr(module, 'block_size', 0)
            gate_post = getattr(module, 'gate_post', None)
            gate_pre = getattr(module, 'gate_pre', None)

            replacement = WPHLinear(
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
                print(f"[WPH] Layer {name}: extra features "
                      f"(gate_post={gate_post is not None}, gate_pre={gate_pre is not None}, "
                      f"correction={correction_A is not None}, block_mixing={block_mixing is not None})")

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            n_converted += 1

    if verbose:
        print(f"[WPH] Converted {n_converted} layers, skipped {n_skipped} (quantize_enabled=False)")

    return model, n_converted, n_skipped
