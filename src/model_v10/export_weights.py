import struct
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

from .config import ModelConfig
from .model import ModelV2
from .components.ternary import TernaryLinear, DEFAULT_THRESHOLD_COEFF
from .components.ssm import SelectiveSSM, SelectiveSSMSSD
from .components.attention import SlidingWindowAttention


TRNM_MAGIC = b'TRNM'
TRNM_VERSION = 1
TRNM_HEADER_SIZE = 256
TRNM_ALIGN = 64
TRNM_MAX_SWA_LAYERS = 32

TRNM_LAYER_SSM = 0
TRNM_LAYER_SWA = 1

TRNM_FLAG_TERNARY = 1 << 0
TRNM_FLAG_HAS_LM_HEAD = 1 << 1
TRNM_FLAG_LM_TERNARY = 1 << 2
TRNM_FLAG_TSCS = 1 << 3
TRNM_FLAG_SSD = 1 << 4
TRNM_FLAG_RELU = 1 << 5
TRNM_FLAG_BITSHIFT_NORM = 1 << 6

TRNM_LFLAG_HAS_ROUTER = 1 << 0
TRNM_LFLAG_HAS_MOE = 1 << 1
TRNM_LFLAG_IS_LTI = 1 << 2


def _build_crc32c_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x82F63B78
            else:
                crc >>= 1
        table.append(crc)
    return table


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC32C_TABLE[(crc ^ b) & 0xFF]
    return crc ^ 0xFFFFFFFF


def align_pos(f, alignment=TRNM_ALIGN):
    pos = f.tell()
    aligned = (pos + alignment - 1) & ~(alignment - 1)
    if aligned > pos:
        f.write(b'\x00' * (aligned - pos))


def pack_ternary_2bit(q_values: torch.Tensor) -> bytes:
    q_values = q_values.cpu().numpy().astype('int8')
    packed = bytearray()

    for i in range(0, len(q_values), 4):
        chunk = q_values[i:i+4]
        byte_val = 0
        for j, q in enumerate(chunk):
            if q == 0:
                bits = 0b00
            elif q == 1:
                bits = 0b01
            elif q == -1:
                bits = 0b10
            else:
                bits = 0b00
            byte_val |= (bits << (2 * j))
        packed.append(byte_val)

    return bytes(packed)


def quantize_ternary_linear(weight: torch.Tensor, group_size: int = 128) -> Tuple[bytes, torch.Tensor]:
    orig_shape = weight.shape
    out_features, in_features = orig_shape

    flat = weight.reshape(-1)
    pad_len = (group_size - flat.numel() % group_size) % group_size
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))

    grouped = flat.reshape(-1, group_size)
    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales
    abs_normalized = normalized.abs()
    threshold = DEFAULT_THRESHOLD_COEFF * abs_normalized.mean(dim=-1, keepdim=True)
    quantized = normalized.sign() * (abs_normalized > threshold).float()

    packed = pack_ternary_2bit(quantized.reshape(-1)[:weight.numel()])

    scales = scales.squeeze(-1).float()

    return packed, scales


def write_float_array(f, tensor: torch.Tensor) -> int:
    start = f.tell()
    data = tensor.cpu().float().numpy().tobytes()
    f.write(data)
    align_pos(f, TRNM_ALIGN)
    return start


def write_ternary_block(f, weight: torch.Tensor, group_size: int) -> Tuple[int, int, int]:
    header_pos = f.tell()
    out_features, in_features = weight.shape

    packed, scales = quantize_ternary_linear(weight, group_size)
    packed_size = len(packed)

    header = struct.pack('<IIII', out_features, in_features, packed_size, 0)
    f.write(header)
    align_pos(f, TRNM_ALIGN)

    data_pos = f.tell()
    f.write(packed)
    align_pos(f, TRNM_ALIGN)

    scales_pos = f.tell()
    scales_bytes = scales.cpu().float().numpy().tobytes()
    f.write(scales_bytes)
    align_pos(f, TRNM_ALIGN)

    return header_pos, data_pos, scales_pos


def write_int8_array(f, tensor: torch.Tensor) -> int:
    start = f.tell()
    data = tensor.cpu().to(torch.int8).numpy().tobytes()
    f.write(data)
    align_pos(f, TRNM_ALIGN)
    return start


def export_model(model: ModelV2, config: ModelConfig, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.cpu()

    flags = 0
    if config.use_ternary:
        flags |= TRNM_FLAG_TERNARY
    if model.lm_head is not None:
        flags |= TRNM_FLAG_HAS_LM_HEAD
        if isinstance(model.lm_head, TernaryLinear):
            flags |= TRNM_FLAG_LM_TERNARY
    if config.use_tscs:
        flags |= TRNM_FLAG_TSCS
    if config.use_ssd:
        flags |= TRNM_FLAG_SSD
    if getattr(config, 'activation_fn', 'silu') == 'relu':
        flags |= TRNM_FLAG_RELU
    if getattr(config, 'use_bitshift_norm', False):
        flags |= TRNM_FLAG_BITSHIFT_NORM

    d_inner = config.expand * config.d_model
    mlp_hidden = int(config.d_model * 2.67)
    head_dim = config.d_model // config.n_heads
    rope_dim = head_dim

    swa_indices = list(config.swa_layer_indices) + [0] * (TRNM_MAX_SWA_LAYERS - len(config.swa_layer_indices))

    header_data = struct.pack(
        '<4sIIIIIIIIIIIIIIIIIIQQQQQ' + 'I' * TRNM_MAX_SWA_LAYERS + 'II',
        TRNM_MAGIC,
        TRNM_VERSION,
        flags,
        config.n_layer,
        config.vocab_size,
        config.d_model,
        config.d_state,
        config.d_conv,
        config.expand,
        config.dt_rank,
        config.n_heads,
        config.ssd_n_heads if config.use_ssd else config.n_heads,
        config.window_size,
        config.seq_len,
        config.ternary_group_size,
        len(config.swa_layer_indices),
        d_inner,
        mlp_hidden,
        head_dim,
        rope_dim,
        0,
        0,
        0,
        0,
        0,
        *swa_indices,
        0,
        0
    )

    if len(header_data) > TRNM_HEADER_SIZE - 4:
        raise ValueError(f"Header data too large: {len(header_data)} > {TRNM_HEADER_SIZE - 4}")

    header_data += b'\x00' * (TRNM_HEADER_SIZE - 4 - len(header_data))

    with open(output_path, 'wb') as f:
        f.write(header_data)

        placeholder_layer_table_offset = f.tell()
        layer_table_placeholder = b'\x00' * (24 * config.n_layer)
        f.write(layer_table_placeholder)
        align_pos(f, TRNM_ALIGN)

        embedding_offset = f.tell()
        embedding_bytes = model.embedding.weight.cpu().float().numpy().tobytes()
        f.write(embedding_bytes)
        align_pos(f, TRNM_ALIGN)

        layer_offsets = []
        layer_entries = []

        for i, layer in enumerate(model.layers):
            is_swa = config.is_swa_layer(i)
            layer_type = TRNM_LAYER_SWA if is_swa else TRNM_LAYER_SSM
            layer_flags = 0

            if hasattr(layer, 'router') and layer.router is not None:
                layer_flags |= TRNM_LFLAG_HAS_ROUTER
            if hasattr(layer, 'moe_ffn') and layer.moe_ffn is not None:
                layer_flags |= TRNM_LFLAG_HAS_MOE
            if is_swa:
                pass
            else:
                if hasattr(layer.block, 'lti_mode') and layer.block.lti_mode:
                    layer_flags |= TRNM_LFLAG_IS_LTI

            layer_data_start = f.tell()

            norm_bytes = layer.norm.weight.cpu().float().numpy().tobytes()
            f.write(norm_bytes)
            align_pos(f, TRNM_ALIGN)

            if is_swa:
                swa_block = layer.block
                attn_norm_bytes = swa_block.attn_norm.weight.cpu().float().numpy().tobytes()
                f.write(attn_norm_bytes)
                align_pos(f, TRNM_ALIGN)

                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    proj = getattr(swa_block, proj_name)
                    if isinstance(proj, TernaryLinear):
                        write_ternary_block(f, proj.weight, config.ternary_group_size)
                    else:
                        weight_bytes = proj.weight.cpu().float().numpy().tobytes()
                        f.write(weight_bytes)
                        align_pos(f, TRNM_ALIGN)

                mlp_norm_bytes = swa_block.mlp_norm.weight.cpu().float().numpy().tobytes()
                f.write(mlp_norm_bytes)
                align_pos(f, TRNM_ALIGN)

                for proj_name in ['mlp_gate', 'mlp_up', 'mlp_down']:
                    proj = getattr(swa_block, proj_name)
                    if isinstance(proj, TernaryLinear):
                        write_ternary_block(f, proj.weight, config.ternary_group_size)
                    else:
                        weight_bytes = proj.weight.cpu().float().numpy().tobytes()
                        f.write(weight_bytes)
                        align_pos(f, TRNM_ALIGN)

                rope_inv_freq_bytes = swa_block.rotary.inv_freq.cpu().float().numpy().tobytes()
                f.write(rope_inv_freq_bytes)
                align_pos(f, TRNM_ALIGN)
            else:
                ssm_block = layer.block
                in_proj = ssm_block.in_proj
                if isinstance(in_proj, TernaryLinear):
                    write_ternary_block(f, in_proj.weight, config.ternary_group_size)
                else:
                    weight_bytes = in_proj.weight.cpu().float().numpy().tobytes()
                    f.write(weight_bytes)
                    align_pos(f, TRNM_ALIGN)

                conv_bytes = ssm_block.conv1d.weight.cpu().float().numpy().tobytes()
                f.write(conv_bytes)
                align_pos(f, TRNM_ALIGN)

                if hasattr(ssm_block, 'lti_mode') and not ssm_block.lti_mode:
                    x_proj = ssm_block.x_proj
                    if isinstance(x_proj, TernaryLinear):
                        write_ternary_block(f, x_proj.weight, config.ternary_group_size)
                    else:
                        weight_bytes = x_proj.weight.cpu().float().numpy().tobytes()
                        f.write(weight_bytes)
                        align_pos(f, TRNM_ALIGN)

                    dt_proj_bytes = ssm_block.dt_proj.weight.cpu().float().numpy().tobytes()
                    f.write(dt_proj_bytes)
                    align_pos(f, TRNM_ALIGN)

                    if hasattr(ssm_block.dt_proj, 'bias'):
                        dt_bias_bytes = ssm_block.dt_proj.bias.cpu().float().numpy().tobytes()
                        f.write(dt_bias_bytes)
                        align_pos(f, TRNM_ALIGN)

                if config.use_tscs and hasattr(ssm_block, 'A_shadow'):
                    a_shadow_bytes = ssm_block.A_shadow.cpu().float().numpy().tobytes()
                    f.write(a_shadow_bytes)
                    align_pos(f, TRNM_ALIGN)
                else:
                    a_log_bytes = ssm_block.A_log.cpu().float().numpy().tobytes()
                    f.write(a_log_bytes)
                    align_pos(f, TRNM_ALIGN)

                d_bytes = ssm_block.D.cpu().float().numpy().tobytes()
                f.write(d_bytes)
                align_pos(f, TRNM_ALIGN)

                out_proj = ssm_block.out_proj
                if isinstance(out_proj, TernaryLinear):
                    write_ternary_block(f, out_proj.weight, config.ternary_group_size)
                else:
                    weight_bytes = out_proj.weight.cpu().float().numpy().tobytes()
                    f.write(weight_bytes)
                    align_pos(f, TRNM_ALIGN)

            layer_data_end = f.tell()
            data_size = layer_data_end - layer_data_start
            data_crc = crc32c(b'\x00' * data_size)

            layer_entry = struct.pack('<BBHI QI I',
                layer_type,
                layer_flags,
                0,
                data_size,
                layer_data_start,
                data_crc,
                0
            )
            layer_entries.append(layer_entry)

        final_norm_offset = f.tell()
        final_norm_bytes = model.final_norm.weight.cpu().float().numpy().tobytes()
        f.write(final_norm_bytes)
        align_pos(f, TRNM_ALIGN)

        lm_head_offset = 0
        if model.lm_head is not None:
            lm_head_offset = f.tell()
            if isinstance(model.lm_head, TernaryLinear):
                write_ternary_block(f, model.lm_head.weight, config.ternary_group_size)
            else:
                weight_bytes = model.lm_head.weight.cpu().float().numpy().tobytes()
                f.write(weight_bytes)
                align_pos(f, TRNM_ALIGN)

        file_size = f.tell()

        f.seek(TRNM_HEADER_SIZE - 32)
        f.write(struct.pack('<QQQQ', file_size, placeholder_layer_table_offset, embedding_offset, final_norm_offset))
        f.write(struct.pack('<Q', lm_head_offset))

        header_crc = crc32c(header_data[:-4])
        f.write(struct.pack('<I', header_crc))

        f.seek(placeholder_layer_table_offset)
        for entry in layer_entries:
            f.write(entry)


def load_config(config_name: str) -> ModelConfig:
    from . import config as config_module

    config_func = getattr(config_module, f'config_{config_name}', None)
    if config_func is None:
        raise ValueError(f"Unknown config: {config_name}")

    return config_func()


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to TRNM binary format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path for TRNM file')
    parser.add_argument('--config', type=str, default='ternary_6m', help='Config name (default: ternary_6m)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to load model on (default: cpu)')

    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    print(f"Creating model...")
    model = ModelV2(config)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Exporting to TRNM format: {args.output}")
    export_model(model, config, args.output)

    output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Export complete! File size: {output_size_mb:.2f} MB")


if __name__ == '__main__':
    main()
