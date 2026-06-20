#!/usr/bin/env python3
"""Lift an R426 smaller-width v12 checkpoint into a full-width template.

The v12 checkpoint loader intentionally rejects dimension mismatches. This tool
keeps that safety property: it starts from a valid full-width checkpoint
template, copies overlapping trained tensors from a smaller checkpoint, and
writes a full-width checkpoint that the normal loader can validate.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = 0x43323156
VERSION = 8
K_HOT = 4000


@dataclass(frozen=True)
class Dims:
    n_layers: int
    d_model: int
    d_state: int
    vocab_size: int
    conv_kernel: int
    mtp_rank: int
    d_embed: int

    @property
    def header(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.n_layers,
            self.d_model,
            self.d_state,
            self.vocab_size,
            self.conv_kernel,
            self.mtp_rank,
        )


@dataclass(frozen=True)
class Entry:
    name: str
    offset: int
    shape: tuple[int, ...]
    elem_size: int = 4

    @property
    def byte_count(self) -> int:
        n = 1
        for v in self.shape:
            n *= v
        return n * self.elem_size


@dataclass
class ParsedCheckpoint:
    path: Path
    blob: bytes
    dims_offset: int
    train_state: bytes
    entries: dict[str, Entry]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-d-model", type=int, default=512)
    parser.add_argument("--source-d-state", type=int, default=64)
    parser.add_argument("--source-d-embed", type=int, default=64)
    parser.add_argument("--target-d-model", type=int, default=1024)
    parser.add_argument("--target-d-state", type=int, default=128)
    parser.add_argument("--target-d-embed", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32064)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--mtp-rank", type=int, default=20)
    parser.add_argument("--copy-train-state", action="store_true", default=True)
    return parser.parse_args()


def find_dims_offset(blob: bytes, dims: Dims) -> int:
    needle = struct.pack("<6I", *dims.header)
    idx = blob.find(needle, 8)
    if idx < 0:
        raise ValueError(f"dimension header not found: {dims.header}")
    return idx


def read_u8(blob: bytes, offset: int) -> int:
    return blob[offset]


def add(entries: dict[str, Entry], name: str, offset: int, shape: tuple[int, ...], elem_size: int = 4) -> int:
    entry = Entry(name=name, offset=offset, shape=shape, elem_size=elem_size)
    entries[name] = entry
    return offset + entry.byte_count


def parse_checkpoint(path: Path, dims: Dims) -> ParsedCheckpoint:
    blob = path.read_bytes()
    if len(blob) < 32:
        raise ValueError(f"{path} is too small to be a v12 checkpoint")
    magic, version = struct.unpack_from("<2I", blob, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"{path} has magic/version {(magic, version)}, expected {(MAGIC, VERSION)}")

    dims_offset = find_dims_offset(blob, dims)
    train_state = blob[8:dims_offset]
    offset = dims_offset + 24
    d = dims.d_model
    ds = dims.d_state
    de = dims.d_embed
    v = dims.vocab_size
    k_conv = dims.conv_kernel
    r = dims.mtp_rank
    entries: dict[str, Entry] = {}

    for layer in range(dims.n_layers):
        prefix = f"layers.{layer}"
        offset = add(entries, f"{prefix}.A", offset, (ds,))
        offset = add(entries, f"{prefix}.B", offset, (ds,))
        offset = add(entries, f"{prefix}.C", offset, (ds, ds))
        offset = add(entries, f"{prefix}.shadow_w_in", offset, (d, ds))
        offset = add(entries, f"{prefix}.sampled_w_in", offset, (d, ds))
        offset = add(entries, f"{prefix}.shadow_w_out", offset, (ds, d))
        offset = add(entries, f"{prefix}.sampled_w_out", offset, (ds, d))
        offset = add(entries, f"{prefix}.shadow_w_conv", offset, (k_conv, ds))

        has_mtp = read_u8(blob, offset)
        offset = add(entries, f"{prefix}.has_mtp", offset, (1,), elem_size=1)
        if has_mtp:
            offset = add(entries, f"{prefix}.mtp_U", offset, (ds, r))
            offset = add(entries, f"{prefix}.mtp_V", offset, (ds, r))

        offset = add(entries, f"{prefix}.m_w_in_row", offset, (d,))
        offset = add(entries, f"{prefix}.m_w_in_col", offset, (ds,))
        offset = add(entries, f"{prefix}.v_w_in_row", offset, (d,))
        offset = add(entries, f"{prefix}.v_w_in_col", offset, (ds,))
        offset = add(entries, f"{prefix}.m_w_out_row", offset, (ds,))
        offset = add(entries, f"{prefix}.m_w_out_col", offset, (d,))
        offset = add(entries, f"{prefix}.v_w_out_row", offset, (ds,))
        offset = add(entries, f"{prefix}.v_w_out_col", offset, (d,))
        offset = add(entries, f"{prefix}.m_A", offset, (ds,))
        offset = add(entries, f"{prefix}.v_A", offset, (ds,))
        offset = add(entries, f"{prefix}.m_B", offset, (ds,))
        offset = add(entries, f"{prefix}.v_B", offset, (ds,))

        has_c_moments = read_u8(blob, offset)
        offset = add(entries, f"{prefix}.has_C_moments", offset, (1,), elem_size=1)
        if has_c_moments:
            offset = add(entries, f"{prefix}.m_C", offset, (ds, ds))
            offset = add(entries, f"{prefix}.v_C", offset, (ds, ds))

        offset = add(entries, f"{prefix}.m_conv", offset, (k_conv, ds))
        offset = add(entries, f"{prefix}.v_conv", offset, (k_conv, ds))

        has_mtp_m = read_u8(blob, offset)
        offset = add(entries, f"{prefix}.has_mtp_m", offset, (1,), elem_size=1)
        if has_mtp_m:
            offset = add(entries, f"{prefix}.m_mtp_U", offset, (ds, r))
            offset = add(entries, f"{prefix}.v_mtp_U", offset, (ds, r))
            offset = add(entries, f"{prefix}.m_mtp_V", offset, (ds, r))
            offset = add(entries, f"{prefix}.v_mtp_V", offset, (ds, r))

        offset = add(entries, f"{prefix}.W_A", offset, (ds, ds))
        offset = add(entries, f"{prefix}.b_A", offset, (ds,))
        offset = add(entries, f"{prefix}.m_W_A", offset, (ds, ds))
        offset = add(entries, f"{prefix}.v_W_A", offset, (ds, ds))
        offset = add(entries, f"{prefix}.m_b_A", offset, (ds,))
        offset = add(entries, f"{prefix}.v_b_A", offset, (ds,))

    offset = add(entries, "shadow_lm_head", offset, (v, d))
    offset = add(entries, "sampled_lm_head", offset, (v, d))
    offset = add(entries, "shadow_w_pred", offset, (d, d))
    offset = add(entries, "sampled_w_pred", offset, (d, d))
    offset = add(entries, "norm_w", offset, (d,))
    offset = add(entries, "norm_b", offset, (d,))
    offset = add(entries, "embed_E", offset, (v, de))
    offset = add(entries, "embed_P", offset, (de, d))
    offset = add(entries, "m_embed_E", offset, (v, de))
    offset = add(entries, "v_embed_E", offset, (v, de))
    offset = add(entries, "m_embed_P", offset, (de, d))
    offset = add(entries, "v_embed_P", offset, (de, d))
    offset = add(entries, "tau_mux", offset, (dims.n_layers,))
    offset = add(entries, "m_lm_head_row", offset, (K_HOT,))
    offset = add(entries, "m_lm_head_col", offset, (d,))
    offset = add(entries, "v_lm_head_row", offset, (K_HOT,))
    offset = add(entries, "v_lm_head_col", offset, (d,))
    offset = add(entries, "m_w_pred_row", offset, (d,))
    offset = add(entries, "m_w_pred_col", offset, (d,))
    offset = add(entries, "v_w_pred_row", offset, (d,))
    offset = add(entries, "v_w_pred_col", offset, (d,))
    offset = add(entries, "m_norm_w", offset, (d,))
    offset = add(entries, "v_norm_w", offset, (d,))
    offset = add(entries, "m_norm_b", offset, (d,))
    offset = add(entries, "v_norm_b", offset, (d,))

    if offset != len(blob):
        raise ValueError(f"{path} parse ended at {offset}, file size is {len(blob)}")

    return ParsedCheckpoint(path=path, blob=blob, dims_offset=dims_offset, train_state=train_state, entries=entries)


def copy_overlap(src_blob: bytes, dst_blob: bytearray, src: Entry, dst: Entry) -> tuple[int, int]:
    if src.elem_size != dst.elem_size:
        raise ValueError(f"element-size mismatch for {src.name}: {src.elem_size} vs {dst.elem_size}")
    if len(src.shape) != len(dst.shape):
        raise ValueError(f"rank mismatch for {src.name}: {src.shape} vs {dst.shape}")

    elem = src.elem_size
    if len(src.shape) == 1:
        n = min(src.shape[0], dst.shape[0])
        dst_blob[dst.offset:dst.offset + n * elem] = src_blob[src.offset:src.offset + n * elem]
        return n, n

    if len(src.shape) == 2:
        rows = min(src.shape[0], dst.shape[0])
        cols = min(src.shape[1], dst.shape[1])
        src_cols = src.shape[1]
        dst_cols = dst.shape[1]
        for row in range(rows):
            src_pos = src.offset + ((row * src_cols) * elem)
            dst_pos = dst.offset + ((row * dst_cols) * elem)
            dst_blob[dst_pos:dst_pos + cols * elem] = src_blob[src_pos:src_pos + cols * elem]
        return rows, cols

    raise ValueError(f"unsupported rank for {src.name}: {src.shape}")


def lift_checkpoint(args: argparse.Namespace) -> None:
    src_dims = Dims(
        n_layers=args.n_layers,
        d_model=args.source_d_model,
        d_state=args.source_d_state,
        vocab_size=args.vocab_size,
        conv_kernel=args.conv_kernel,
        mtp_rank=args.mtp_rank,
        d_embed=args.source_d_embed,
    )
    dst_dims = Dims(
        n_layers=args.n_layers,
        d_model=args.target_d_model,
        d_state=args.target_d_state,
        vocab_size=args.vocab_size,
        conv_kernel=args.conv_kernel,
        mtp_rank=args.mtp_rank,
        d_embed=args.target_d_embed,
    )

    source = parse_checkpoint(args.source, src_dims)
    template = parse_checkpoint(args.template, dst_dims)
    if len(source.train_state) != len(template.train_state):
        raise ValueError(
            f"TrainState size mismatch: source={len(source.train_state)} template={len(template.train_state)}"
        )

    output = bytearray(template.blob)
    if args.copy_train_state:
        output[8:template.dims_offset] = source.train_state

    copied = 0
    skipped_flags = 0
    for name, src_entry in source.entries.items():
        dst_entry = template.entries.get(name)
        if dst_entry is None:
            continue
        if src_entry.elem_size == 1:
            if source.blob[src_entry.offset:src_entry.offset + src_entry.byte_count] != template.blob[dst_entry.offset:dst_entry.offset + dst_entry.byte_count]:
                raise ValueError(f"presence flag mismatch for {name}")
            skipped_flags += 1
            continue
        copy_overlap(source.blob, output, src_entry, dst_entry)
        copied += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_bytes(output)
    shutil.move(str(tmp), str(args.output))

    print(
        "[R426_CHECKPOINT_LIFT] active=1 "
        f"source={args.source} template={args.template} output={args.output} "
        f"source_dims={src_dims.d_model}/{src_dims.d_state}/{src_dims.d_embed} "
        f"target_dims={dst_dims.d_model}/{dst_dims.d_state}/{dst_dims.d_embed} "
        f"train_state_bytes={len(source.train_state)} copied_entries={copied} "
        f"validated_flags={skipped_flags} status=ok"
    )


def main() -> int:
    args = parse_args()
    lift_checkpoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
