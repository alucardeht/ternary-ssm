"""T017 packed linear Triton kernel gate."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .footprint import BYTES_PER_GIB
from .t004_real_qwen_bridge import find_weight_dir, inventory_safetensors, resolve_qwen_config
from .t006_real_residency_block_smoke import BlockCodecPolicy
from .t008_compressed_generation_smoke import _load_weight_map
from .t016_persistent_packed_cache import (
    PackedCacheStats,
    _cached_binary_payload,
    _cached_payload,
    _default_cache_dir,
    _pack_tensor,
    _payload_cache_key,
    _payload_nbytes,
    _payload_with_trit_codes,
    _reconstruct_packed,
)
from .t015_packed_qwen_inference import (
    _FallbackTokenizer,
    _apply_rope,
    _cuda_memory,
    _read_rows,
    _read_tensor,
    _rms_norm,
    _select_device,
    _silu,
    _t014_a_denominator,
    _tokenizer,
)
from .t020_device_credit_flow import run_t020_device_credit_flow_gate
from .t020_packed_work_coalescing import run_t020_packed_work_coalescing_gate


ACTIVATION = "T017_PACKED_LINEAR_KERNEL_ACTIVE"


@dataclass(frozen=True)
class PackedLinearKernelReceipt:
    backend: str
    rows: int
    cols: int
    tokens: int
    repair_count: int
    iterations: int
    max_abs_error: float
    relative_rmse: float
    reference_seconds_per_iter: float
    kernel_seconds_per_iter: float
    speedup_vs_reconstruct_matmul: float
    output_sha256: str

    @property
    def passes(self) -> bool:
        return (
            self.backend == "triton"
            and self.max_abs_error <= 2.0e-3
            and self.relative_rmse <= 2.0e-4
            and self.speedup_vs_reconstruct_matmul > 1.0
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passes"] = self.passes
        return data


@dataclass(frozen=True)
class KernelInferenceReceipt:
    prompt_text: str
    tokenizer_source: str
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    generated_text: str
    requested_layers: int
    completed_layers: int
    device: str
    elapsed_seconds: float
    tokens_per_second: float
    kernel_seconds: float
    peak_cuda_allocated_gib: float | None
    peak_cuda_reserved_gib: float | None
    peak_process_working_set_gib: float | None
    process_working_set_gib: float | None
    process_private_usage_gib: float | None
    denominator_total_gib: float
    denominator_slack_gib: float
    cache_stats: dict[str, Any]
    output_sha256: str
    resident_payload_cache: bool = False
    preload_payloads: bool = False
    preload_seconds: float = 0.0
    device_payload_cache: bool = False
    device_payload_budget_gib: float | None = None
    preload_device_payloads: bool = False
    dense_tensor_cache: bool = False
    trit_packed_codes: bool = False
    device_credit_flow: dict[str, Any] | None = None
    packed_work_coalescing: dict[str, Any] | None = None
    fail_closed_device_payloads: bool = False
    device_payload_descriptor_table: dict[str, Any] | None = None
    hot_payload_load_seconds: float = 0.0
    hot_payload_load_events: int = 0
    defer_kernel_synchronization: bool = False
    gpu_argmax_head: bool = False
    kernel_timing_mode: str = "synchronized"
    packed_linear_block_m: int = 16
    packed_linear_block_c: int = 128
    packed_linear_block_r: int = 32
    fused_mlp_gate_up_kernel: bool = False
    fused_qkv_compact_projection_kernel: bool = False
    prompt_embeddings_preloaded: bool = False
    prompt_embedding_preload_seconds: float = 0.0
    native_attention_sdp: bool = False
    trit_code_group_size: int = 4
    t031_lm_head_certificate: dict[str, Any] | None = None
    t034_device_argmax_embedding_splice: dict[str, Any] | None = None

    @property
    def passes(self) -> bool:
        return (
            self.completed_layers == self.requested_layers
            and self.requested_layers > 0
            and len(self.generated_token_ids) > 0
            and bool(self.generated_text)
            and self.denominator_total_gib <= 12.0
            and self.elapsed_seconds > 0.0
            and self.kernel_seconds > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_token_ids"] = list(self.generated_token_ids)
        data["passes"] = self.passes
        return data


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


if _triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _trit_base3_divisor(code_digit):
        return tl.where(
            code_digit == 0,
            1,
            tl.where(
                code_digit == 1,
                3,
                tl.where(
                    code_digit == 2,
                    9,
                    tl.where(
                        code_digit == 3,
                        27,
                        tl.where(
                            code_digit == 4,
                            81,
                            tl.where(
                                code_digit == 5,
                                243,
                                tl.where(
                                    code_digit == 6,
                                    729,
                                    tl.where(
                                        code_digit == 7,
                                        2187,
                                        tl.where(
                                            code_digit == 8,
                                            6561,
                                            tl.where(
                                                code_digit == 9,
                                                19683,
                                                tl.where(
                                                    code_digit == 10,
                                                    59049,
                                                    tl.where(
                                                        code_digit == 11,
                                                        177147,
                                                        tl.where(
                                                            code_digit == 12,
                                                            531441,
                                                            tl.where(
                                                                code_digit == 13,
                                                                1594323,
                                                                tl.where(
                                                                    code_digit == 14,
                                                                    4782969,
                                                                    tl.where(
                                                                        code_digit == 15,
                                                                        14348907,
                                                                        tl.where(
                                                                            code_digit == 16,
                                                                            43046721,
                                                                            tl.where(
                                                                                code_digit == 17,
                                                                                129140163,
                                                                                tl.where(
                                                                                    code_digit == 18,
                                                                                    387420489,
                                                                                    1162261467,
                                                                                ),
                                                                            ),
                                                                        ),
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    @triton.jit
    def _packed_linear_kernel(
        x_ptr,
        codes_ptr,
        scales_ptr,
        repair_indices_ptr,
        repair_values_ptr,
        out_ptr,
        tokens: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        repair_count: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ) -> None:
        pid_s = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

        for c0 in range(0, cols, BLOCK_C):
            c = c0 + offs_c
            x_mask = (offs_s[:, None] < tokens) & (c[None, :] < cols)
            code_mask = (offs_m[:, None] < rows) & (c[None, :] < cols)
            x_vals = tl.load(x_ptr + offs_s[:, None] * cols + c[None, :], mask=x_mask, other=0.0)
            code_vals = tl.load(codes_ptr + offs_m[:, None] * cols + c[None, :], mask=code_mask, other=0).to(tl.float32)
            acc += tl.dot(x_vals, tl.trans(code_vals), input_precision="ieee")

        scales = tl.load(scales_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
        acc = acc * scales[None, :]

        offs_r = tl.arange(0, BLOCK_R)
        for r0 in range(0, repair_count, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows) & (r[None, :] < repair_count)
            idx = tl.load(
                repair_indices_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                repair_values_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0.0,
            ).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x_repair * vals[None, :, :], axis=2)

        out_mask = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows)
        tl.store(out_ptr + offs_s[:, None] * rows + offs_m[None, :], acc, mask=out_mask)

    @triton.jit
    def _packed_linear_residual_kernel(
        x_ptr,
        residual_ptr,
        codes_ptr,
        scales_ptr,
        repair_indices_ptr,
        repair_values_ptr,
        out_ptr,
        tokens: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        repair_count: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ) -> None:
        pid_s = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

        for c0 in range(0, cols, BLOCK_C):
            c = c0 + offs_c
            x_mask = (offs_s[:, None] < tokens) & (c[None, :] < cols)
            code_mask = (offs_m[:, None] < rows) & (c[None, :] < cols)
            x_vals = tl.load(x_ptr + offs_s[:, None] * cols + c[None, :], mask=x_mask, other=0.0)
            code_vals = tl.load(codes_ptr + offs_m[:, None] * cols + c[None, :], mask=code_mask, other=0).to(tl.float32)
            acc += tl.dot(x_vals, tl.trans(code_vals), input_precision="ieee")

        scales = tl.load(scales_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
        acc = acc * scales[None, :]

        offs_r = tl.arange(0, BLOCK_R)
        for r0 in range(0, repair_count, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows) & (r[None, :] < repair_count)
            idx = tl.load(
                repair_indices_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                repair_values_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0.0,
            ).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x_repair * vals[None, :, :], axis=2)

        out_mask = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows)
        residual = tl.load(residual_ptr + offs_s[:, None] * rows + offs_m[None, :], mask=out_mask, other=0.0)
        tl.store(out_ptr + offs_s[:, None] * rows + offs_m[None, :], acc + residual, mask=out_mask)

    @triton.jit
    def _packed_linear_trit_kernel(
        x_ptr,
        trit_codes_ptr,
        scales_ptr,
        repair_indices_ptr,
        repair_values_ptr,
        out_ptr,
        tokens: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        trit_cols: tl.constexpr,
        TRIT_GROUP_SIZE: tl.constexpr,
        repair_count: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ) -> None:
        pid_s = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

        for c0 in range(0, cols, BLOCK_C):
            c = c0 + offs_c
            x_mask = (offs_s[:, None] < tokens) & (c[None, :] < cols)
            code_group = c // TRIT_GROUP_SIZE
            code_digit = c - code_group * TRIT_GROUP_SIZE
            divisor = _trit_base3_divisor(code_digit)
            code_mask = (offs_m[:, None] < rows) & (c[None, :] < cols) & (code_group[None, :] < trit_cols)
            x_vals = tl.load(x_ptr + offs_s[:, None] * cols + c[None, :], mask=x_mask, other=0.0)
            packed_vals = tl.load(
                trit_codes_ptr + offs_m[:, None] * trit_cols + code_group[None, :],
                mask=code_mask,
                other=13,
            ).to(tl.int32)
            code_vals = (((packed_vals // divisor[None, :]) % 3) - 1).to(tl.float32)
            acc += tl.dot(x_vals, tl.trans(code_vals), input_precision="ieee")

        scales = tl.load(scales_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
        acc = acc * scales[None, :]

        offs_r = tl.arange(0, BLOCK_R)
        for r0 in range(0, repair_count, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows) & (r[None, :] < repair_count)
            idx = tl.load(
                repair_indices_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                repair_values_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0.0,
            ).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x_repair * vals[None, :, :], axis=2)

        out_mask = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows)
        tl.store(out_ptr + offs_s[:, None] * rows + offs_m[None, :], acc, mask=out_mask)

    @triton.jit
    def _packed_linear_trit_residual_kernel(
        x_ptr,
        residual_ptr,
        trit_codes_ptr,
        scales_ptr,
        repair_indices_ptr,
        repair_values_ptr,
        out_ptr,
        tokens: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        trit_cols: tl.constexpr,
        TRIT_GROUP_SIZE: tl.constexpr,
        repair_count: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ) -> None:
        pid_s = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

        for c0 in range(0, cols, BLOCK_C):
            c = c0 + offs_c
            x_mask = (offs_s[:, None] < tokens) & (c[None, :] < cols)
            code_group = c // TRIT_GROUP_SIZE
            code_digit = c - code_group * TRIT_GROUP_SIZE
            divisor = _trit_base3_divisor(code_digit)
            code_mask = (offs_m[:, None] < rows) & (c[None, :] < cols) & (code_group[None, :] < trit_cols)
            x_vals = tl.load(x_ptr + offs_s[:, None] * cols + c[None, :], mask=x_mask, other=0.0)
            packed_vals = tl.load(
                trit_codes_ptr + offs_m[:, None] * trit_cols + code_group[None, :],
                mask=code_mask,
                other=13,
            ).to(tl.int32)
            code_vals = (((packed_vals // divisor[None, :]) % 3) - 1).to(tl.float32)
            acc += tl.dot(x_vals, tl.trans(code_vals), input_precision="ieee")

        scales = tl.load(scales_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
        acc = acc * scales[None, :]

        offs_r = tl.arange(0, BLOCK_R)
        for r0 in range(0, repair_count, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows) & (r[None, :] < repair_count)
            idx = tl.load(
                repair_indices_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                repair_values_ptr + offs_m[:, None] * repair_count + r[None, :],
                mask=repair_mask,
                other=0.0,
            ).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x_repair * vals[None, :, :], axis=2)

        out_mask = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows)
        residual = tl.load(residual_ptr + offs_s[:, None] * rows + offs_m[None, :], mask=out_mask, other=0.0)
        tl.store(out_ptr + offs_s[:, None] * rows + offs_m[None, :], acc + residual, mask=out_mask)

    @triton.jit
    def _packed_linear_trit_pair_kernel(
        x_ptr,
        trit_codes_a_ptr,
        scales_a_ptr,
        repair_indices_a_ptr,
        repair_values_a_ptr,
        out_a_ptr,
        trit_codes_b_ptr,
        scales_b_ptr,
        repair_indices_b_ptr,
        repair_values_b_ptr,
        out_b_ptr,
        tokens: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        trit_cols_a: tl.constexpr,
        trit_cols_b: tl.constexpr,
        TRIT_GROUP_SIZE: tl.constexpr,
        repair_count_a: tl.constexpr,
        repair_count_b: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ) -> None:
        pid_s = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        acc_a = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)
        acc_b = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

        for c0 in range(0, cols, BLOCK_C):
            c = c0 + offs_c
            x_mask = (offs_s[:, None] < tokens) & (c[None, :] < cols)
            code_group = c // TRIT_GROUP_SIZE
            code_digit = c - code_group * TRIT_GROUP_SIZE
            divisor = _trit_base3_divisor(code_digit)
            x_vals = tl.load(x_ptr + offs_s[:, None] * cols + c[None, :], mask=x_mask, other=0.0)
            code_mask_a = (offs_m[:, None] < rows) & (c[None, :] < cols) & (code_group[None, :] < trit_cols_a)
            packed_a = tl.load(
                trit_codes_a_ptr + offs_m[:, None] * trit_cols_a + code_group[None, :],
                mask=code_mask_a,
                other=13,
            ).to(tl.int32)
            code_a = (((packed_a // divisor[None, :]) % 3) - 1).to(tl.float32)
            acc_a += tl.dot(x_vals, tl.trans(code_a), input_precision="ieee")
            code_mask_b = (offs_m[:, None] < rows) & (c[None, :] < cols) & (code_group[None, :] < trit_cols_b)
            packed_b = tl.load(
                trit_codes_b_ptr + offs_m[:, None] * trit_cols_b + code_group[None, :],
                mask=code_mask_b,
                other=13,
            ).to(tl.int32)
            code_b = (((packed_b // divisor[None, :]) % 3) - 1).to(tl.float32)
            acc_b += tl.dot(x_vals, tl.trans(code_b), input_precision="ieee")

        scales_a = tl.load(scales_a_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
        scales_b = tl.load(scales_b_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
        acc_a = acc_a * scales_a[None, :]
        acc_b = acc_b * scales_b[None, :]

        offs_r = tl.arange(0, BLOCK_R)
        for r0 in range(0, repair_count_a, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows) & (r[None, :] < repair_count_a)
            idx = tl.load(
                repair_indices_a_ptr + offs_m[:, None] * repair_count_a + r[None, :],
                mask=repair_mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                repair_values_a_ptr + offs_m[:, None] * repair_count_a + r[None, :],
                mask=repair_mask,
                other=0.0,
            ).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc_a += tl.sum(x_repair * vals[None, :, :], axis=2)

        for r0 in range(0, repair_count_b, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows) & (r[None, :] < repair_count_b)
            idx = tl.load(
                repair_indices_b_ptr + offs_m[:, None] * repair_count_b + r[None, :],
                mask=repair_mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                repair_values_b_ptr + offs_m[:, None] * repair_count_b + r[None, :],
                mask=repair_mask,
                other=0.0,
            ).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc_b += tl.sum(x_repair * vals[None, :, :], axis=2)

        out_mask = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows)
        tl.store(out_a_ptr + offs_s[:, None] * rows + offs_m[None, :], acc_a, mask=out_mask)
        tl.store(out_b_ptr + offs_s[:, None] * rows + offs_m[None, :], acc_b, mask=out_mask)

    @triton.jit
    def _packed_linear_trit_triple_kernel(
        x_ptr,
        trit_codes_a_ptr,
        scales_a_ptr,
        repair_indices_a_ptr,
        repair_values_a_ptr,
        out_a_ptr,
        trit_codes_b_ptr,
        scales_b_ptr,
        repair_indices_b_ptr,
        repair_values_b_ptr,
        out_b_ptr,
        trit_codes_c_ptr,
        scales_c_ptr,
        repair_indices_c_ptr,
        repair_values_c_ptr,
        out_c_ptr,
        tokens: tl.constexpr,
        rows_a: tl.constexpr,
        rows_b: tl.constexpr,
        rows_c: tl.constexpr,
        cols: tl.constexpr,
        trit_cols_a: tl.constexpr,
        trit_cols_b: tl.constexpr,
        trit_cols_c: tl.constexpr,
        TRIT_GROUP_SIZE: tl.constexpr,
        repair_count_a: tl.constexpr,
        repair_count_b: tl.constexpr,
        repair_count_c: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ) -> None:
        pid_s = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, BLOCK_C)
        acc_a = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)
        acc_b = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)
        acc_c = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

        for c0 in range(0, cols, BLOCK_C):
            c = c0 + offs_c
            x_mask = (offs_s[:, None] < tokens) & (c[None, :] < cols)
            code_group = c // TRIT_GROUP_SIZE
            code_digit = c - code_group * TRIT_GROUP_SIZE
            divisor = _trit_base3_divisor(code_digit)
            x_vals = tl.load(x_ptr + offs_s[:, None] * cols + c[None, :], mask=x_mask, other=0.0)

            code_mask_a = (offs_m[:, None] < rows_a) & (c[None, :] < cols) & (code_group[None, :] < trit_cols_a)
            packed_a = tl.load(
                trit_codes_a_ptr + offs_m[:, None] * trit_cols_a + code_group[None, :],
                mask=code_mask_a,
                other=13,
            ).to(tl.int32)
            code_a = (((packed_a // divisor[None, :]) % 3) - 1).to(tl.float32)
            acc_a += tl.dot(x_vals, tl.trans(code_a), input_precision="ieee")

            code_mask_b = (offs_m[:, None] < rows_b) & (c[None, :] < cols) & (code_group[None, :] < trit_cols_b)
            packed_b = tl.load(
                trit_codes_b_ptr + offs_m[:, None] * trit_cols_b + code_group[None, :],
                mask=code_mask_b,
                other=13,
            ).to(tl.int32)
            code_b = (((packed_b // divisor[None, :]) % 3) - 1).to(tl.float32)
            acc_b += tl.dot(x_vals, tl.trans(code_b), input_precision="ieee")

            code_mask_c = (offs_m[:, None] < rows_c) & (c[None, :] < cols) & (code_group[None, :] < trit_cols_c)
            packed_c = tl.load(
                trit_codes_c_ptr + offs_m[:, None] * trit_cols_c + code_group[None, :],
                mask=code_mask_c,
                other=13,
            ).to(tl.int32)
            code_c = (((packed_c // divisor[None, :]) % 3) - 1).to(tl.float32)
            acc_c += tl.dot(x_vals, tl.trans(code_c), input_precision="ieee")

        scales_a = tl.load(scales_a_ptr + offs_m, mask=offs_m < rows_a, other=0.0).to(tl.float32)
        scales_b = tl.load(scales_b_ptr + offs_m, mask=offs_m < rows_b, other=0.0).to(tl.float32)
        scales_c = tl.load(scales_c_ptr + offs_m, mask=offs_m < rows_c, other=0.0).to(tl.float32)
        acc_a = acc_a * scales_a[None, :]
        acc_b = acc_b * scales_b[None, :]
        acc_c = acc_c * scales_c[None, :]

        offs_r = tl.arange(0, BLOCK_R)
        for r0 in range(0, repair_count_a, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows_a) & (r[None, :] < repair_count_a)
            idx = tl.load(repair_indices_a_ptr + offs_m[:, None] * repair_count_a + r[None, :], mask=repair_mask, other=0).to(tl.int32)
            vals = tl.load(repair_values_a_ptr + offs_m[:, None] * repair_count_a + r[None, :], mask=repair_mask, other=0.0).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc_a += tl.sum(x_repair * vals[None, :, :], axis=2)

        for r0 in range(0, repair_count_b, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows_b) & (r[None, :] < repair_count_b)
            idx = tl.load(repair_indices_b_ptr + offs_m[:, None] * repair_count_b + r[None, :], mask=repair_mask, other=0).to(tl.int32)
            vals = tl.load(repair_values_b_ptr + offs_m[:, None] * repair_count_b + r[None, :], mask=repair_mask, other=0.0).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc_b += tl.sum(x_repair * vals[None, :, :], axis=2)

        for r0 in range(0, repair_count_c, BLOCK_R):
            r = r0 + offs_r
            repair_mask = (offs_m[:, None] < rows_c) & (r[None, :] < repair_count_c)
            idx = tl.load(repair_indices_c_ptr + offs_m[:, None] * repair_count_c + r[None, :], mask=repair_mask, other=0).to(tl.int32)
            vals = tl.load(repair_values_c_ptr + offs_m[:, None] * repair_count_c + r[None, :], mask=repair_mask, other=0.0).to(tl.float32)
            x_repair = tl.load(
                x_ptr + offs_s[:, None, None] * cols + idx[None, :, :],
                mask=(offs_s[:, None, None] < tokens) & repair_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc_c += tl.sum(x_repair * vals[None, :, :], axis=2)

        out_mask_a = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows_a)
        out_mask_b = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows_b)
        out_mask_c = (offs_s[:, None] < tokens) & (offs_m[None, :] < rows_c)
        tl.store(out_a_ptr + offs_s[:, None] * rows_a + offs_m[None, :], acc_a, mask=out_mask_a)
        tl.store(out_b_ptr + offs_s[:, None] * rows_b + offs_m[None, :], acc_b, mask=out_mask_b)
        tl.store(out_c_ptr + offs_s[:, None] * rows_c + offs_m[None, :], acc_c, mask=out_mask_c)


def packed_linear_triton(
    x: Any,
    payload: dict[str, Any],
    *,
    residual: Any | None = None,
    block_s: int = 4,
    block_m: int = 16,
    block_c: int = 128,
    block_r: int = 32,
) -> Any:
    """Compute x @ W.T directly from the T016 packed payload."""

    if not _triton_available():
        raise RuntimeError("triton_missing")
    if not x.is_cuda:
        raise RuntimeError("packed_linear_triton_requires_cuda_input")
    import torch
    import triton

    x_contig = x.contiguous().float()
    rows, cols = [int(dim) for dim in payload["shape"]]
    if int(x_contig.shape[1]) != cols:
        raise ValueError(f"x cols {int(x_contig.shape[1])} != payload cols {cols}")
    residual_contig = None
    if residual is not None:
        residual_contig = residual.contiguous().float()
        if tuple(residual_contig.shape) != (int(x_contig.shape[0]), rows):
            raise ValueError(f"residual shape {tuple(residual_contig.shape)} != {(int(x_contig.shape[0]), rows)}")
    use_trit_codes = "trit_codes" in payload
    codes = None
    trit_codes = None
    trit_cols = 0
    if use_trit_codes:
        trit_codes = _tensor_on_device(payload["trit_codes"], device=x_contig.device, dtype=payload["trit_codes"].dtype)
        trit_cols = int(trit_codes.shape[1])
    else:
        codes = _tensor_on_device(payload["codes"], device=x_contig.device, dtype=torch.int8)
    trit_group_size = int(payload.get("trit_group_size", 4))
    scales = _tensor_on_device(payload["scales"], device=x_contig.device, dtype=payload["scales"].dtype)
    repair_indices = _tensor_on_device(
        payload["repair_indices"],
        device=x_contig.device,
        dtype=payload["repair_indices"].dtype,
    )
    repair_values = _tensor_on_device(
        payload["repair_values"],
        device=x_contig.device,
        dtype=payload["repair_values"].dtype,
    )
    repair_count = int(repair_indices.shape[1])
    out = torch.empty((int(x_contig.shape[0]), rows), device=x_contig.device, dtype=torch.float32)
    grid = (triton.cdiv(int(x_contig.shape[0]), block_s), triton.cdiv(rows, block_m))
    if use_trit_codes and residual_contig is None:
        _packed_linear_trit_kernel[grid](
            x_contig,
            trit_codes,
            scales,
            repair_indices,
            repair_values,
            out,
            int(x_contig.shape[0]),
            rows,
            cols,
            trit_cols,
            trit_group_size,
            repair_count,
            BLOCK_S=block_s,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            BLOCK_R=block_r,
        )
    elif use_trit_codes:
        _packed_linear_trit_residual_kernel[grid](
            x_contig,
            residual_contig,
            trit_codes,
            scales,
            repair_indices,
            repair_values,
            out,
            int(x_contig.shape[0]),
            rows,
            cols,
            trit_cols,
            trit_group_size,
            repair_count,
            BLOCK_S=block_s,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            BLOCK_R=block_r,
        )
    elif residual_contig is None:
        _packed_linear_kernel[grid](
            x_contig,
            codes,
            scales,
            repair_indices,
            repair_values,
            out,
            int(x_contig.shape[0]),
            rows,
            cols,
            repair_count,
            BLOCK_S=block_s,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            BLOCK_R=block_r,
        )
    else:
        _packed_linear_residual_kernel[grid](
            x_contig,
            residual_contig,
            codes,
            scales,
            repair_indices,
            repair_values,
            out,
            int(x_contig.shape[0]),
            rows,
            cols,
            repair_count,
            BLOCK_S=block_s,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            BLOCK_R=block_r,
        )
    return out


def packed_linear_triton_pair(
    x: Any,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    block_s: int = 4,
    block_m: int = 16,
    block_c: int = 128,
    block_r: int = 32,
) -> tuple[Any, Any]:
    """Compute two trit-packed affine projections sharing the same input read."""

    if "trit_codes" not in payload_a or "trit_codes" not in payload_b:
        return (
            packed_linear_triton(x, payload_a, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
            packed_linear_triton(x, payload_b, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
        )
    trit_group_size_a = int(payload_a.get("trit_group_size", 4))
    trit_group_size_b = int(payload_b.get("trit_group_size", 4))
    if trit_group_size_a != trit_group_size_b:
        return (
            packed_linear_triton(x, payload_a, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
            packed_linear_triton(x, payload_b, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
        )
    if not _triton_available():
        raise RuntimeError("triton_missing")
    if not x.is_cuda:
        raise RuntimeError("packed_linear_triton_pair_requires_cuda_input")
    import torch
    import triton

    x_contig = x.contiguous().float()
    rows_a, cols_a = [int(dim) for dim in payload_a["shape"]]
    rows_b, cols_b = [int(dim) for dim in payload_b["shape"]]
    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError(f"paired payload shapes differ: {(rows_a, cols_a)} != {(rows_b, cols_b)}")
    if int(x_contig.shape[1]) != cols_a:
        raise ValueError(f"x cols {int(x_contig.shape[1])} != payload cols {cols_a}")
    trit_codes_a = _tensor_on_device(payload_a["trit_codes"], device=x_contig.device, dtype=payload_a["trit_codes"].dtype)
    trit_codes_b = _tensor_on_device(payload_b["trit_codes"], device=x_contig.device, dtype=payload_b["trit_codes"].dtype)
    scales_a = _tensor_on_device(payload_a["scales"], device=x_contig.device, dtype=payload_a["scales"].dtype)
    scales_b = _tensor_on_device(payload_b["scales"], device=x_contig.device, dtype=payload_b["scales"].dtype)
    repair_indices_a = _tensor_on_device(
        payload_a["repair_indices"],
        device=x_contig.device,
        dtype=payload_a["repair_indices"].dtype,
    )
    repair_indices_b = _tensor_on_device(
        payload_b["repair_indices"],
        device=x_contig.device,
        dtype=payload_b["repair_indices"].dtype,
    )
    repair_values_a = _tensor_on_device(
        payload_a["repair_values"],
        device=x_contig.device,
        dtype=payload_a["repair_values"].dtype,
    )
    repair_values_b = _tensor_on_device(
        payload_b["repair_values"],
        device=x_contig.device,
        dtype=payload_b["repair_values"].dtype,
    )
    out_a = torch.empty((int(x_contig.shape[0]), rows_a), device=x_contig.device, dtype=torch.float32)
    out_b = torch.empty((int(x_contig.shape[0]), rows_b), device=x_contig.device, dtype=torch.float32)
    grid = (triton.cdiv(int(x_contig.shape[0]), block_s), triton.cdiv(rows_a, block_m))
    _packed_linear_trit_pair_kernel[grid](
        x_contig,
        trit_codes_a,
        scales_a,
        repair_indices_a,
        repair_values_a,
        out_a,
        trit_codes_b,
        scales_b,
        repair_indices_b,
        repair_values_b,
        out_b,
        int(x_contig.shape[0]),
        rows_a,
        cols_a,
        int(trit_codes_a.shape[1]),
        int(trit_codes_b.shape[1]),
        trit_group_size_a,
        int(repair_indices_a.shape[1]),
        int(repair_indices_b.shape[1]),
        BLOCK_S=block_s,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        BLOCK_R=block_r,
    )
    return out_a, out_b


def packed_linear_triton_triple(
    x: Any,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    payload_c: dict[str, Any],
    *,
    block_s: int = 4,
    block_m: int = 16,
    block_c: int = 128,
    block_r: int = 32,
) -> tuple[Any, Any, Any]:
    """Compute three trit-packed affine projections sharing the same input read."""

    if "trit_codes" not in payload_a or "trit_codes" not in payload_b or "trit_codes" not in payload_c:
        return (
            packed_linear_triton(x, payload_a, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
            packed_linear_triton(x, payload_b, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
            packed_linear_triton(x, payload_c, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
        )
    trit_group_size_a = int(payload_a.get("trit_group_size", 4))
    trit_group_size_b = int(payload_b.get("trit_group_size", 4))
    trit_group_size_c = int(payload_c.get("trit_group_size", 4))
    if trit_group_size_a != trit_group_size_b or trit_group_size_a != trit_group_size_c:
        return (
            packed_linear_triton(x, payload_a, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
            packed_linear_triton(x, payload_b, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
            packed_linear_triton(x, payload_c, block_s=block_s, block_m=block_m, block_c=block_c, block_r=block_r),
        )
    if not _triton_available():
        raise RuntimeError("triton_missing")
    if not x.is_cuda:
        raise RuntimeError("packed_linear_triton_triple_requires_cuda_input")
    import torch
    import triton

    x_contig = x.contiguous().float()
    rows_a, cols_a = [int(dim) for dim in payload_a["shape"]]
    rows_b, cols_b = [int(dim) for dim in payload_b["shape"]]
    rows_c, cols_c = [int(dim) for dim in payload_c["shape"]]
    if cols_a != cols_b or cols_a != cols_c:
        raise ValueError(f"triple payload input widths differ: {cols_a}, {cols_b}, {cols_c}")
    if int(x_contig.shape[1]) != cols_a:
        raise ValueError(f"x cols {int(x_contig.shape[1])} != payload cols {cols_a}")

    trit_codes_a = _tensor_on_device(payload_a["trit_codes"], device=x_contig.device, dtype=payload_a["trit_codes"].dtype)
    trit_codes_b = _tensor_on_device(payload_b["trit_codes"], device=x_contig.device, dtype=payload_b["trit_codes"].dtype)
    trit_codes_c = _tensor_on_device(payload_c["trit_codes"], device=x_contig.device, dtype=payload_c["trit_codes"].dtype)
    scales_a = _tensor_on_device(payload_a["scales"], device=x_contig.device, dtype=payload_a["scales"].dtype)
    scales_b = _tensor_on_device(payload_b["scales"], device=x_contig.device, dtype=payload_b["scales"].dtype)
    scales_c = _tensor_on_device(payload_c["scales"], device=x_contig.device, dtype=payload_c["scales"].dtype)
    repair_indices_a = _tensor_on_device(
        payload_a["repair_indices"], device=x_contig.device, dtype=payload_a["repair_indices"].dtype
    )
    repair_indices_b = _tensor_on_device(
        payload_b["repair_indices"], device=x_contig.device, dtype=payload_b["repair_indices"].dtype
    )
    repair_indices_c = _tensor_on_device(
        payload_c["repair_indices"], device=x_contig.device, dtype=payload_c["repair_indices"].dtype
    )
    repair_values_a = _tensor_on_device(
        payload_a["repair_values"], device=x_contig.device, dtype=payload_a["repair_values"].dtype
    )
    repair_values_b = _tensor_on_device(
        payload_b["repair_values"], device=x_contig.device, dtype=payload_b["repair_values"].dtype
    )
    repair_values_c = _tensor_on_device(
        payload_c["repair_values"], device=x_contig.device, dtype=payload_c["repair_values"].dtype
    )
    out_a = torch.empty((int(x_contig.shape[0]), rows_a), device=x_contig.device, dtype=torch.float32)
    out_b = torch.empty((int(x_contig.shape[0]), rows_b), device=x_contig.device, dtype=torch.float32)
    out_c = torch.empty((int(x_contig.shape[0]), rows_c), device=x_contig.device, dtype=torch.float32)
    max_rows = max(rows_a, rows_b, rows_c)
    grid = (triton.cdiv(int(x_contig.shape[0]), block_s), triton.cdiv(max_rows, block_m))
    _packed_linear_trit_triple_kernel[grid](
        x_contig,
        trit_codes_a,
        scales_a,
        repair_indices_a,
        repair_values_a,
        out_a,
        trit_codes_b,
        scales_b,
        repair_indices_b,
        repair_values_b,
        out_b,
        trit_codes_c,
        scales_c,
        repair_indices_c,
        repair_values_c,
        out_c,
        int(x_contig.shape[0]),
        rows_a,
        rows_b,
        rows_c,
        cols_a,
        int(trit_codes_a.shape[1]),
        int(trit_codes_b.shape[1]),
        int(trit_codes_c.shape[1]),
        trit_group_size_a,
        int(repair_indices_a.shape[1]),
        int(repair_indices_b.shape[1]),
        int(repair_indices_c.shape[1]),
        BLOCK_S=block_s,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        BLOCK_R=block_r,
    )
    return out_a, out_b, out_c


def _tensor_on_device(tensor: Any, *, device: Any, dtype: Any) -> Any:
    if tensor.device == device and tensor.dtype == dtype and tensor.is_contiguous():
        return tensor
    return tensor.to(device=device, dtype=dtype).contiguous()


def _payload_to_device(payload: dict[str, Any], *, device: str) -> dict[str, Any]:
    import torch

    torch_device = torch.device(device)
    moved = dict(payload)
    if "trit_codes" in payload:
        moved["trit_codes"] = _tensor_on_device(
            payload["trit_codes"],
            device=torch_device,
            dtype=payload["trit_codes"].dtype,
        )
        moved.pop("codes", None)
    else:
        moved["codes"] = _tensor_on_device(payload["codes"], device=torch_device, dtype=torch.int8)
    moved["scales"] = _tensor_on_device(payload["scales"], device=torch_device, dtype=payload["scales"].dtype)
    moved["repair_indices"] = _tensor_on_device(
        payload["repair_indices"],
        device=torch_device,
        dtype=payload["repair_indices"].dtype,
    )
    moved["repair_values"] = _tensor_on_device(
        payload["repair_values"],
        device=torch_device,
        dtype=payload["repair_values"].dtype,
    )
    return moved


def _device_payload_descriptor(
    *,
    key: str,
    name: str,
    start: int | None,
    stop: int | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    tensors: dict[str, dict[str, Any]] = {}
    for tensor_name in ("trit_codes", "codes", "scales", "repair_indices", "repair_values"):
        tensor = payload.get(tensor_name)
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            tensors[tensor_name] = {
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "device": str(tensor.device),
                "bytes": int(tensor.numel()) * int(tensor.element_size()),
            }
    t031_structural = _t031_structural_payload_descriptor(payload)
    return {
        "key": key,
        "name": name,
        "start": start,
        "stop": stop,
        "shape": [int(dim) for dim in payload["shape"]],
        "policy": payload["policy"],
        "trit_packed_codes": "trit_codes" in payload,
        "trit_group_size": int(payload.get("trit_group_size", 0) or 0),
        "bytes": sum(item["bytes"] for item in tensors.values()),
        "tensors": tensors,
        "t031_structural_payload": t031_structural,
    }


def _t031_structural_payload_descriptor(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    trit_codes = payload.get("trit_codes")
    codes = payload.get("codes")
    scales = payload.get("scales")
    repair_values = payload.get("repair_values")
    rows = int((payload.get("shape") or (0, 0))[0])
    zero_code_rows = 0
    zero_code_groups = 0
    code_groups = 0
    if trit_codes is not None:
        group_size = int(payload.get("trit_group_size", 4))
        zero_group = sum(3**index for index in range(group_size))
        trit_tensor = trit_codes.detach()
        zero_mask = trit_tensor == zero_group
        zero_code_groups = int(zero_mask.sum().item())
        code_groups = int(trit_tensor.numel())
        zero_code_rows = int(zero_mask.all(dim=1).sum().item()) if trit_tensor.ndim == 2 else 0
    elif codes is not None:
        codes_tensor = codes.detach()
        zero_mask = codes_tensor == 0
        zero_code_groups = int(zero_mask.sum().item())
        code_groups = int(codes_tensor.numel())
        zero_code_rows = int(zero_mask.all(dim=1).sum().item()) if codes_tensor.ndim == 2 else 0

    repair_free_rows = 0
    if repair_values is not None and rows > 0:
        repair_tensor = repair_values.detach()
        repair_free_rows = int((repair_tensor == 0).all(dim=1).sum().item()) if repair_tensor.ndim == 2 else 0

    scale_chamber_count = 0
    scale_index_bytes = 0
    if scales is not None:
        scale_tensor = scales.detach().contiguous()
        scale_chamber_count = int(torch.unique(scale_tensor).numel())
        if rows > 0:
            scale_index_bytes = rows if scale_chamber_count <= 256 else rows * 2

    return {
        "zero_code_rows": zero_code_rows,
        "zero_code_groups": zero_code_groups,
        "code_groups": code_groups,
        "zero_group_fraction": zero_code_groups / code_groups if code_groups > 0 else 0.0,
        "repair_free_rows": repair_free_rows,
        "repair_free_fraction": repair_free_rows / rows if rows > 0 else 0.0,
        "scale_chamber_count": scale_chamber_count,
        "scale_index_bytes": scale_index_bytes,
        "row_count": rows,
        "byte_owner_classes": [
            "trit_codes",
            "scales",
            "repair_indices",
            "repair_values",
            "descriptor_metadata",
        ],
    }


def _trit_payload_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "trit-payloads" / cache_key


def _load_trit_payload(cache_dir: Path, cache_key: str, stats: PackedCacheStats) -> dict[str, Any] | None:
    import torch

    path = _trit_payload_cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    started = time.perf_counter()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stats.trit_payloads_loaded += 1
    stats.trit_cache_bytes_read += path.stat().st_size
    stats.load_seconds += time.perf_counter() - started
    return payload


def _write_trit_payload(cache_dir: Path, cache_key: str, payload: dict[str, Any], stats: PackedCacheStats) -> None:
    import torch

    started = time.perf_counter()
    path = _trit_payload_cache_path(cache_dir, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    stats.trit_payloads_written += 1
    stats.trit_cache_bytes_written += path.stat().st_size
    stats.pack_seconds += time.perf_counter() - started


def _payload_from_cache_or_resident(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    name: str,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache: dict[str, dict[str, Any]] | None = None,
    device_payload_cache_device: str | None = None,
    device_payload_cache_budget_bytes: int | None = None,
    binary_cache: bool = False,
    trit_packed_codes: bool = False,
    trit_code_group_size: int = 4,
    start: int | None = None,
    stop: int | None = None,
    fail_closed_device_payloads: bool = False,
    device_payload_descriptors: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    load_payload = _cached_binary_payload if binary_cache else _cached_payload
    key = _payload_cache_key(
        model_dir,
        weight_map,
        name,
        target_mlp_bits=target_mlp_bits,
        start=start,
        stop=stop,
    )
    if device_payload_cache is not None and key in device_payload_cache:
        stats.device_resident_hits += 1
        if fail_closed_device_payloads:
            stats.device_arena_hits += 1
        return device_payload_cache[key]
    if fail_closed_device_payloads:
        stats.device_arena_misses += 1
        raise RuntimeError(f"device_payload_arena_miss:{name}:{start}:{stop}")
    payload: dict[str, Any] | None = None
    use_cached_trit_payload = trit_packed_codes and trit_code_group_size == 4
    if use_cached_trit_payload and not refresh_cache:
        payload = _load_trit_payload(cache_dir, key, stats)
        if payload is not None:
            stats.trit_payloads += 1
            stats.trit_code_bytes += int(payload.get("trit_code_bytes", 0))
            stats.unpacked_code_bytes_avoided += int(payload.get("unpacked_code_bytes", 0))

    if payload is None:
        if payload_cache is None:
            payload = load_payload(
                model_dir=model_dir,
                weight_map=weight_map,
                cache_dir=cache_dir,
                stats=stats,
                name=name,
                target_mlp_bits=target_mlp_bits,
                start=start,
                stop=stop,
                refresh_cache=refresh_cache,
            )
        elif key in payload_cache:
            stats.resident_hits += 1
            payload = payload_cache[key]
        else:
            payload = load_payload(
                model_dir=model_dir,
                weight_map=weight_map,
                cache_dir=cache_dir,
                stats=stats,
                name=name,
                target_mlp_bits=target_mlp_bits,
                start=start,
                stop=stop,
                refresh_cache=refresh_cache,
            )
            payload_cache[key] = payload
            stats.resident_stores += 1
            stats.resident_bytes += _payload_nbytes(payload)

    if trit_packed_codes and (
        "trit_codes" not in payload or int(payload.get("trit_group_size", 4)) != int(trit_code_group_size)
    ):
        payload = _payload_with_trit_codes(payload, drop_unpacked=True, group_size=int(trit_code_group_size))
        if trit_code_group_size == 4:
            _write_trit_payload(cache_dir, key, payload, stats)
        stats.trit_payloads += 1
        stats.trit_code_bytes += int(payload.get("trit_code_bytes", 0))
        stats.unpacked_code_bytes_avoided += int(payload.get("unpacked_code_bytes", 0))

    if device_payload_cache is not None and device_payload_cache_device is not None:
        payload_bytes = _payload_nbytes(payload)
        if (
            device_payload_cache_budget_bytes is not None
            and stats.device_resident_bytes + payload_bytes > device_payload_cache_budget_bytes
        ):
            stats.device_resident_skips += 1
            return payload
        try:
            device_payload = _payload_to_device(payload, device=device_payload_cache_device)
            device_payload_cache[key] = device_payload
            if device_payload_descriptors is not None:
                device_payload_descriptors[key] = _device_payload_descriptor(
                    key=key,
                    name=name,
                    start=start,
                    stop=stop,
                    payload=device_payload,
                )
            stats.device_resident_stores += 1
            stats.device_resident_bytes += _payload_nbytes(device_payload)
            return device_payload
        except RuntimeError as exc:
            stats.device_resident_failures += 1
            if "out of memory" in str(exc).lower():
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return payload
    return payload


def _kernel_linear_from_cache(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    name: str,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None = None,
    device_payload_cache: dict[str, dict[str, Any]] | None = None,
    device_payload_cache_device: str | None = None,
    device_payload_cache_budget_bytes: int | None = None,
    binary_cache: bool = False,
    trit_packed_codes: bool = False,
    trit_code_group_size: int = 4,
    start: int | None = None,
    stop: int | None = None,
    residual: Any | None = None,
    fail_closed_device_payloads: bool = False,
    device_payload_descriptors: dict[str, dict[str, Any]] | None = None,
    synchronize_kernel: bool = True,
    packed_linear_block_m: int = 16,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
) -> tuple[Any, float]:
    import torch

    payload = _payload_from_cache_or_resident(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name,
        target_mlp_bits=target_mlp_bits,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        start=start,
        stop=stop,
        refresh_cache=refresh_cache,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
    )
    if synchronize_kernel:
        torch.cuda.synchronize()
    started = time.perf_counter()
    out = packed_linear_triton(
        x,
        payload,
        residual=residual,
        block_m=packed_linear_block_m,
        block_c=packed_linear_block_c,
        block_r=packed_linear_block_r,
    )
    if synchronize_kernel:
        torch.cuda.synchronize()
    return out, time.perf_counter() - started


def _kernel_linear_pair_from_cache(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    name_a: str,
    name_b: str,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None = None,
    device_payload_cache: dict[str, dict[str, Any]] | None = None,
    device_payload_cache_device: str | None = None,
    device_payload_cache_budget_bytes: int | None = None,
    binary_cache: bool = False,
    trit_packed_codes: bool = False,
    trit_code_group_size: int = 4,
    fail_closed_device_payloads: bool = False,
    device_payload_descriptors: dict[str, dict[str, Any]] | None = None,
    synchronize_kernel: bool = True,
    packed_linear_block_m: int = 16,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
) -> tuple[Any, Any, float]:
    import torch

    payload_a = _payload_from_cache_or_resident(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name_a,
        target_mlp_bits=target_mlp_bits,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        refresh_cache=refresh_cache,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
    )
    payload_b = _payload_from_cache_or_resident(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name_b,
        target_mlp_bits=target_mlp_bits,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        refresh_cache=refresh_cache,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
    )
    if synchronize_kernel:
        torch.cuda.synchronize()
    started = time.perf_counter()
    out_a, out_b = packed_linear_triton_pair(
        x,
        payload_a,
        payload_b,
        block_m=packed_linear_block_m,
        block_c=packed_linear_block_c,
        block_r=packed_linear_block_r,
    )
    if synchronize_kernel:
        torch.cuda.synchronize()
    return out_a, out_b, time.perf_counter() - started


def _kernel_linear_triple_from_cache(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    name_a: str,
    name_b: str,
    name_c: str,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None = None,
    device_payload_cache: dict[str, dict[str, Any]] | None = None,
    device_payload_cache_device: str | None = None,
    device_payload_cache_budget_bytes: int | None = None,
    binary_cache: bool = False,
    trit_packed_codes: bool = False,
    trit_code_group_size: int = 4,
    fail_closed_device_payloads: bool = False,
    device_payload_descriptors: dict[str, dict[str, Any]] | None = None,
    synchronize_kernel: bool = True,
    packed_linear_block_m: int = 16,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
) -> tuple[Any, Any, Any, float]:
    import torch

    payload_a = _payload_from_cache_or_resident(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name_a,
        target_mlp_bits=target_mlp_bits,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        refresh_cache=refresh_cache,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
    )
    payload_b = _payload_from_cache_or_resident(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name_b,
        target_mlp_bits=target_mlp_bits,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        refresh_cache=refresh_cache,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
    )
    payload_c = _payload_from_cache_or_resident(
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=name_c,
        target_mlp_bits=target_mlp_bits,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        refresh_cache=refresh_cache,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
    )
    if synchronize_kernel:
        torch.cuda.synchronize()
    started = time.perf_counter()
    out_a, out_b, out_c = packed_linear_triton_triple(
        x,
        payload_a,
        payload_b,
        payload_c,
        block_m=packed_linear_block_m,
        block_c=packed_linear_block_c,
        block_r=packed_linear_block_r,
    )
    if synchronize_kernel:
        torch.cuda.synchronize()
    return out_a, out_b, out_c, time.perf_counter() - started


def _attention_forward_kernel(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    layer_index: int,
    input_norm: Any,
    q_norm: Any,
    k_norm: Any,
    rms_eps: float,
    rope_theta: float,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache_device: str | None,
    device_payload_cache_budget_bytes: int | None,
    binary_cache: bool,
    trit_packed_codes: bool,
    trit_code_group_size: int,
    packed_work_coalescing: bool,
    fail_closed_device_payloads: bool,
    device_payload_descriptors: dict[str, dict[str, Any]] | None,
    synchronize_kernel: bool,
    packed_linear_block_m: int,
    packed_linear_block_c: int,
    packed_linear_block_r: int,
    native_attention_sdp: bool = False,
    fused_qkv_compact_projection_kernel: bool = False,
) -> tuple[Any, float]:
    import torch

    kernel_seconds = 0.0
    prefix = f"model.layers.{layer_index}.self_attn"
    h = _rms_norm(x, input_norm, rms_eps)
    if fused_qkv_compact_projection_kernel:
        q, k, v, dt = _kernel_linear_triple_from_cache(
            h,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name_a=f"{prefix}.q_proj.weight",
            name_b=f"{prefix}.k_proj.weight",
            name_c=f"{prefix}.v_proj.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
    else:
        q, dt = _kernel_linear_from_cache(
            h,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name=f"{prefix}.q_proj.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
        kernel_seconds += dt
        k, dt = _kernel_linear_from_cache(
            h,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name=f"{prefix}.k_proj.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
        kernel_seconds += dt
        v, dt = _kernel_linear_from_cache(
            h,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name=f"{prefix}.v_proj.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
    kernel_seconds += dt

    head_dim = int(q_norm.shape[0])
    q_heads = int(q.shape[-1] // head_dim)
    kv_heads = int(k.shape[-1] // head_dim)
    seq_len = int(q.shape[0])
    q = q.reshape(seq_len, q_heads, head_dim)
    k = k.reshape(seq_len, kv_heads, head_dim)
    v = v.reshape(seq_len, kv_heads, head_dim)
    q = _rms_norm(q, q_norm, rms_eps)
    k = _rms_norm(k, k_norm, rms_eps)
    positions = torch.arange(seq_len, device=x.device)
    q = _apply_rope(q, theta=rope_theta, positions=positions)
    k = _apply_rope(k, theta=rope_theta, positions=positions)
    repeat = max(1, q_heads // kv_heads)
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    if native_attention_sdp:
        q_sdp = q.permute(1, 0, 2).unsqueeze(0)
        k_sdp = k.permute(1, 0, 2).unsqueeze(0)
        v_sdp = v.permute(1, 0, 2).unsqueeze(0)
        context = torch.nn.functional.scaled_dot_product_attention(q_sdp, k_sdp, v_sdp, is_causal=True)
        context = context.squeeze(0).permute(1, 0, 2).reshape(seq_len, q_heads * head_dim)
    else:
        scores = torch.einsum("shd,thd->hst", q, k) / (head_dim**0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("hst,thd->shd", attn, v).reshape(seq_len, q_heads * head_dim)
    out, dt = _kernel_linear_from_cache(
        context,
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.o_proj.weight",
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        residual=x if packed_work_coalescing else None,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
        synchronize_kernel=synchronize_kernel,
        packed_linear_block_m=packed_linear_block_m,
        packed_linear_block_c=packed_linear_block_c,
        packed_linear_block_r=packed_linear_block_r,
    )
    kernel_seconds += dt
    return (out if packed_work_coalescing else x + out), kernel_seconds


def _mlp_forward_kernel(
    x: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    layer_index: int,
    post_norm: Any,
    rms_eps: float,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache_device: str | None,
    device_payload_cache_budget_bytes: int | None,
    binary_cache: bool,
    trit_packed_codes: bool,
    trit_code_group_size: int,
    packed_work_coalescing: bool,
    inplace_mlp_intermediate: bool = False,
    fail_closed_device_payloads: bool = False,
    device_payload_descriptors: dict[str, dict[str, Any]] | None = None,
    synchronize_kernel: bool = True,
    packed_linear_block_m: int = 16,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
    fused_mlp_gate_up_kernel: bool = False,
) -> tuple[Any, float]:
    kernel_seconds = 0.0
    prefix = f"model.layers.{layer_index}.mlp"
    h = _rms_norm(x, post_norm, rms_eps)
    if fused_mlp_gate_up_kernel:
        gate, up, dt = _kernel_linear_pair_from_cache(
            h,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name_a=f"{prefix}.gate_proj.weight",
            name_b=f"{prefix}.up_proj.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
    else:
        gate, dt = _kernel_linear_from_cache(
            h,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name=f"{prefix}.gate_proj.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
        kernel_seconds += dt
        up, dt = _kernel_linear_from_cache(
            h,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name=f"{prefix}.up_proj.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
    kernel_seconds += dt
    gate = _silu(gate)
    if inplace_mlp_intermediate:
        hidden = gate.mul_(up)
    else:
        hidden = gate * up
    down, dt = _kernel_linear_from_cache(
        hidden,
        model_dir=model_dir,
        weight_map=weight_map,
        cache_dir=cache_dir,
        stats=stats,
        name=f"{prefix}.down_proj.weight",
        target_mlp_bits=target_mlp_bits,
        refresh_cache=refresh_cache,
        payload_cache=payload_cache,
        device_payload_cache=device_payload_cache,
        device_payload_cache_device=device_payload_cache_device,
        device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
        binary_cache=binary_cache,
        trit_packed_codes=trit_packed_codes,
        trit_code_group_size=trit_code_group_size,
        residual=x if packed_work_coalescing else None,
        fail_closed_device_payloads=fail_closed_device_payloads,
        device_payload_descriptors=device_payload_descriptors,
        synchronize_kernel=synchronize_kernel,
        packed_linear_block_m=packed_linear_block_m,
        packed_linear_block_c=packed_linear_block_c,
        packed_linear_block_r=packed_linear_block_r,
    )
    kernel_seconds += dt
    return (down if packed_work_coalescing else x + down), kernel_seconds


def _lm_head_argmax_kernel(
    x_last: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    chunk_rows: int,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache_device: str | None,
    device_payload_cache_budget_bytes: int | None,
    binary_cache: bool,
    trit_packed_codes: bool,
    trit_code_group_size: int,
    fail_closed_device_payloads: bool,
    device_payload_descriptors: dict[str, dict[str, Any]] | None,
    synchronize_kernel: bool,
    gpu_argmax_head: bool,
    t034_device_argmax_embedding_splice: bool,
    embedding_weight: Any | None,
    packed_linear_block_m: int,
    packed_linear_block_c: int,
    packed_linear_block_r: int,
) -> tuple[int, float, dict[str, Any]]:
    import torch

    best_value: float | None = None
    best_index = 0
    best_value_tensor: Any | None = None
    best_index_tensor: Any | None = None
    runner_up_value_tensor: Any | None = None
    kernel_seconds = 0.0
    with safe_open(model_dir / weight_map["lm_head.weight"], framework="pt", device="cpu") as handle:
        rows = int(handle.get_slice("lm_head.weight").get_shape()[0])
    x_one = x_last.reshape(1, -1)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        logits, dt = _kernel_linear_from_cache(
            x_one,
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name="lm_head.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            start=start,
            stop=stop,
            fail_closed_device_payloads=fail_closed_device_payloads,
            device_payload_descriptors=device_payload_descriptors,
            synchronize_kernel=synchronize_kernel,
            packed_linear_block_m=packed_linear_block_m,
            packed_linear_block_c=packed_linear_block_c,
            packed_linear_block_r=packed_linear_block_r,
        )
        kernel_seconds += dt
        value, index = torch.max(logits[0], dim=-1)
        if gpu_argmax_head:
            candidate_index = index.to(dtype=torch.int64) + int(start)
            if best_value_tensor is None or best_index_tensor is None:
                best_value_tensor = value
                best_index_tensor = candidate_index
                runner_up_value_tensor = torch.full_like(value, float("-inf"))
            else:
                better = value > best_value_tensor
                previous_best = best_value_tensor
                best_value_tensor = torch.where(better, value, best_value_tensor)
                best_index_tensor = torch.where(better, candidate_index, best_index_tensor)
                candidate_runner = torch.where(better, previous_best, value)
                runner_up_value_tensor = torch.maximum(runner_up_value_tensor, candidate_runner)
        else:
            if best_value is None or float(value.item()) > best_value:
                best_value = float(value.item())
                best_index = start + int(index.item())
    if gpu_argmax_head:
        if best_index_tensor is None:
            return 0, kernel_seconds, {
                "active": False,
                "reason": "empty_lm_head",
            }
        margin = None
        if runner_up_value_tensor is not None:
            margin = float((best_value_tensor - runner_up_value_tensor).detach().cpu().item())
        splice_certificate: dict[str, Any] = {
            "active": False,
            "reason": "t034_device_argmax_embedding_splice_disabled",
        }
        if t034_device_argmax_embedding_splice:
            if embedding_weight is None:
                splice_certificate = {
                    "active": False,
                    "reason": "embedding_weight_not_resident_on_device",
                    "next_embedding_device_closure_proven": False,
                    "host_token_id_dependency_before_next_embedding": True,
                }
            else:
                token_index_device = best_index_tensor.reshape(1).to(dtype=torch.long)
                next_embedding = embedding_weight.index_select(0, token_index_device).contiguous()
                device_status_word = torch.empty((1,), device=next_embedding.device, dtype=torch.int32)
                host_index = int(best_index_tensor.detach().cpu().item())
                host_reference = embedding_weight[host_index : host_index + 1].contiguous()
                max_abs_error = float(torch.max(torch.abs(next_embedding.float() - host_reference.float())).detach().cpu().item())
                device_status_word.fill_(1 if max_abs_error == 0.0 else 0)
                splice_certificate = {
                    "active": True,
                    "token_index_device": str(token_index_device.device),
                    "embedding_device": str(embedding_weight.device),
                    "next_embedding_device": str(next_embedding.device),
                    "next_embedding_shape": list(next_embedding.shape),
                    "next_embedding_device_closure_proven": max_abs_error == 0.0,
                    "host_token_id_dependency_before_next_embedding": False,
                    "host_token_id_read_after_device_embedding": True,
                    "device_status_word": int(device_status_word.detach().cpu().item()),
                    "max_abs_embedding_error": max_abs_error,
                    "tie_policy": "torch.max first-index inside chunks and strict greater between chunks",
                    "full_logits_materialized": False,
                    "full_logits_returned_to_host": False,
                    "quality_tradeoff": False,
                }
        return int(best_index_tensor.item()), kernel_seconds, {
            "active": True,
            "chunk_rows": int(chunk_rows),
            "device_accumulator": True,
            "winner_margin": margin,
            "certifies_exact_chunked_top1": margin is not None and margin >= 0.0,
            "t033_exact_greedy_output_contract": {
                "active": True,
                "contract": "token_id_only_after_device_reduction",
                "full_logits_materialized": False,
                "full_logits_returned_to_host": False,
                "quality_tradeoff": False,
                "gate_ready": True,
            },
            "t034_device_argmax_embedding_splice": splice_certificate,
            "skipped_work": False,
        }
    return best_index, kernel_seconds, {
        "active": False,
        "reason": "gpu_argmax_head_disabled",
        "t033_exact_greedy_output_contract": {
            "active": False,
            "gate_ready": False,
            "reason": "gpu_argmax_head_disabled",
        },
    }


def _preload_qwen_payloads(
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    cache_dir: Path,
    stats: PackedCacheStats,
    requested_layers: int,
    lm_head_chunk_rows: int,
    target_mlp_bits: float,
    refresh_cache: bool,
    payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache: dict[str, dict[str, Any]] | None,
    device_payload_cache_device: str | None,
    device_payload_cache_budget_bytes: int | None,
    binary_cache: bool,
    trit_packed_codes: bool,
    trit_code_group_size: int,
    device_payload_descriptors: dict[str, dict[str, Any]] | None = None,
) -> float:
    started = time.perf_counter()
    projection_names: list[str] = []
    for layer_index in range(requested_layers):
        attn = f"model.layers.{layer_index}.self_attn"
        mlp = f"model.layers.{layer_index}.mlp"
        projection_names.extend(
            [
                f"{attn}.q_proj.weight",
                f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight",
                f"{attn}.o_proj.weight",
                f"{mlp}.gate_proj.weight",
                f"{mlp}.up_proj.weight",
                f"{mlp}.down_proj.weight",
            ]
        )
    for name in projection_names:
        _payload_from_cache_or_resident(
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name=name,
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            device_payload_descriptors=device_payload_descriptors,
        )
    with safe_open(model_dir / weight_map["lm_head.weight"], framework="pt", device="cpu") as handle:
        rows = int(handle.get_slice("lm_head.weight").get_shape()[0])
    for start in range(0, rows, lm_head_chunk_rows):
        _payload_from_cache_or_resident(
            model_dir=model_dir,
            weight_map=weight_map,
            cache_dir=cache_dir,
            stats=stats,
            name="lm_head.weight",
            target_mlp_bits=target_mlp_bits,
            refresh_cache=refresh_cache,
            payload_cache=payload_cache,
            device_payload_cache=device_payload_cache,
            device_payload_cache_device=device_payload_cache_device,
            device_payload_cache_budget_bytes=device_payload_cache_budget_bytes,
            binary_cache=binary_cache,
            trit_packed_codes=trit_packed_codes,
            trit_code_group_size=trit_code_group_size,
            start=start,
            stop=min(rows, start + lm_head_chunk_rows),
            device_payload_descriptors=device_payload_descriptors,
        )
    return time.perf_counter() - started


def _relative_rmse(actual: Any, expected: Any) -> float:
    import torch

    mse = torch.mean((actual.float() - expected.float()) ** 2)
    denom = torch.mean(expected.float() ** 2).clamp_min(1e-12)
    return float(torch.sqrt(mse / denom).item())


def _time_cuda(fn: Any, *, iterations: int) -> float:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iterations


def run_t017_packed_linear_kernel_gate(
    *,
    rows: int = 512,
    cols: int = 512,
    tokens: int = 4,
    repair_fraction: float = 0.03,
    iterations: int = 25,
    seed: int = 17017,
) -> dict[str, Any]:
    """Run the cheapest packed-linear kernel correctness and speed gate."""

    blockers: list[str] = []
    receipt: PackedLinearKernelReceipt | None = None
    if importlib.util.find_spec("torch") is None:
        blockers.append("torch_missing")
    if not _triton_available():
        blockers.append("triton_missing")
    if not blockers:
        try:
            import hashlib
            import torch

            if not torch.cuda.is_available():
                blockers.append("cuda_missing")
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                exact = torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 0.07
                x = torch.randn((tokens, cols), generator=generator, dtype=torch.float32).cuda()
                policy = BlockCodecPolicy(
                    name="t017_synthetic_kernel_gate",
                    target_bits_per_weight=3.0,
                    repair_fraction=repair_fraction,
                )
                payload = _pack_tensor(exact, policy)
                dense, _ = _reconstruct_packed(payload, device="cuda")
                expected = x @ dense.T
                actual = packed_linear_triton(x, payload)
                torch.cuda.synchronize()
                max_abs_error = float(torch.max(torch.abs(actual - expected)).item())
                relative_rmse = _relative_rmse(actual, expected)

                def reference_once() -> Any:
                    dense_ref, _ = _reconstruct_packed(payload, device="cuda")
                    return x @ dense_ref.T

                def kernel_once() -> Any:
                    return packed_linear_triton(x, payload)

                for _ in range(3):
                    reference_once()
                    kernel_once()
                reference_seconds = _time_cuda(reference_once, iterations=iterations)
                kernel_seconds = _time_cuda(kernel_once, iterations=iterations)
                digest = hashlib.sha256(actual.detach().cpu().numpy().tobytes()).hexdigest()
                receipt = PackedLinearKernelReceipt(
                    backend="triton",
                    rows=rows,
                    cols=cols,
                    tokens=tokens,
                    repair_count=int(payload["repair_indices"].shape[1]),
                    iterations=iterations,
                    max_abs_error=max_abs_error,
                    relative_rmse=relative_rmse,
                    reference_seconds_per_iter=reference_seconds,
                    kernel_seconds_per_iter=kernel_seconds,
                    speedup_vs_reconstruct_matmul=reference_seconds / kernel_seconds if kernel_seconds > 0 else 0.0,
                    output_sha256=digest,
                )
        except Exception as exc:  # pragma: no cover - runtime toolchain guard
            blockers.append(f"t017_kernel_gate_failed:{type(exc).__name__}:{exc}")

    pass_gate = receipt.passes if receipt is not None and not blockers else False
    if receipt is not None and not pass_gate and not blockers:
        blockers.append("t017_kernel_gate_failed_threshold")
    return {
        "probe": "t017_packed_linear_kernel_gate",
        "activation": ACTIVATION,
        "receipt": receipt.as_dict() if receipt is not None else None,
        "readiness": {
            "kernel_gate_pass": pass_gate,
            "backend": receipt.backend if receipt is not None else None,
            "speedup_vs_reconstruct_matmul": receipt.speedup_vs_reconstruct_matmul if receipt is not None else None,
        },
        "blockers": blockers,
        "next_actions": [
            "Integrate packed_linear_triton into T016 layer-1 MLP path."
        ]
        if pass_gate
        else ["Fix the T017 kernel gate blockers before touching all-layer inference."],
    }


def run_t017_kernel_qwen_inference(
    *,
    model_id: str = "Qwen/Qwen3-32B",
    local_model_dir: Path | None = None,
    allow_hf_config_fetch: bool = True,
    prompt_text: str = "Local compressed Qwen smoke:",
    max_new_tokens: int = 1,
    layer_limit: int | None = 1,
    device: str = "auto",
    target_mlp_bits: float = 2.45,
    lm_head_chunk_rows: int = 2048,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    resident_payload_cache: bool = False,
    preload_payloads: bool = False,
    device_payload_cache: bool = False,
    device_payload_budget_gib: float | None = None,
    preload_device_payloads: bool = False,
    dense_tensor_cache: bool = False,
    binary_cache: bool = False,
    trit_packed_codes: bool = False,
    trit_code_group_size: int = 4,
    device_credit_flow: bool = False,
    packed_work_coalescing: bool = False,
    inplace_mlp_intermediate: bool = False,
    fail_closed_device_payloads: bool = False,
    device_payload_descriptor_table: bool = False,
    defer_kernel_synchronization: bool = False,
    gpu_argmax_head: bool = False,
    packed_linear_block_m: int = 16,
    packed_linear_block_c: int = 128,
    packed_linear_block_r: int = 32,
    fused_mlp_gate_up_kernel: bool = False,
    fused_qkv_compact_projection_kernel: bool = False,
    preload_prompt_embeddings: bool = False,
    native_attention_sdp: bool = False,
    t034_device_argmax_embedding_splice: bool = False,
) -> dict[str, Any]:
    """Run Qwen token generation with T017 packed linear kernels."""

    config = resolve_qwen_config(
        model_id=model_id,
        local_model_dir=local_model_dir,
        allow_hf_config_fetch=allow_hf_config_fetch,
    )
    weight_dir = find_weight_dir(config, local_model_dir)
    inventory = inventory_safetensors(weight_dir)
    blockers: list[str] = []
    receipt: KernelInferenceReceipt | None = None
    missing: list[str] = []

    if importlib.util.find_spec("torch") is None:
        blockers.append("torch_missing")
    if not _triton_available():
        blockers.append("triton_missing")
    if not config.matches_project_qwen3_32b_shape():
        blockers.append("config_shape_mismatch")
    if weight_dir is None or not inventory.local_weights_present:
        blockers.append("missing_local_qwen3_32b_safetensors")

    if not blockers and weight_dir is not None:
        try:
            import hashlib
            import json
            import torch

            from .runtime_memory import process_memory_snapshot

            runtime_device = _select_device(device)
            if runtime_device != "cuda":
                blockers.append("t017_kernel_requires_cuda")
            else:
                torch.cuda.reset_peak_memory_stats()
                resolved_cache_dir = cache_dir or _default_cache_dir(weight_dir)
                stats = PackedCacheStats(cache_dir=str(resolved_cache_dir))
                payload_cache: dict[str, dict[str, Any]] | None = {} if resident_payload_cache else None
                device_payloads: dict[str, dict[str, Any]] | None = {} if (device_payload_cache or preload_device_payloads) else None
                device_payload_descriptors: dict[str, dict[str, Any]] | None = {} if device_payload_descriptor_table else None
                device_payload_budget_bytes = (
                    int(float(device_payload_budget_gib) * BYTES_PER_GIB)
                    if device_payload_budget_gib is not None and float(device_payload_budget_gib) > 0.0
                    else None
                )
                dense_cache: dict[str, Any] | None = {} if dense_tensor_cache else None
                weight_map = _load_weight_map(weight_dir)
                total_layers = int(config.layers)
                requested_layers = total_layers if layer_limit is None or layer_limit <= 0 else min(int(layer_limit), total_layers)
                coalescing_gate = (
                    run_t020_packed_work_coalescing_gate(
                        layers=requested_layers,
                        hidden_size=int(config.hidden_size),
                        tokens=max_new_tokens,
                    )
                    if packed_work_coalescing
                    else None
                )
                required = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
                for layer_index in range(requested_layers):
                    prefix = f"model.layers.{layer_index}"
                    required.extend(
                        [
                            f"{prefix}.input_layernorm.weight",
                            f"{prefix}.post_attention_layernorm.weight",
                            f"{prefix}.self_attn.q_norm.weight",
                            f"{prefix}.self_attn.k_norm.weight",
                            f"{prefix}.self_attn.q_proj.weight",
                            f"{prefix}.self_attn.k_proj.weight",
                            f"{prefix}.self_attn.v_proj.weight",
                            f"{prefix}.self_attn.o_proj.weight",
                            f"{prefix}.mlp.gate_proj.weight",
                            f"{prefix}.mlp.up_proj.weight",
                            f"{prefix}.mlp.down_proj.weight",
                        ]
                    )
                missing = [name for name in required if name not in weight_map]
                if missing:
                    blockers.append("missing_t017_kernel_tensors")
                else:
                    try:
                        tokenizer = _tokenizer(weight_dir)
                        tokenizer_source = "transformers_tokenizer"
                    except Exception:
                        tokenizer = _FallbackTokenizer()
                        tokenizer_source = "fallback_byte_tokenizer"
                    token_ids = [int(token) for token in tokenizer.encode(prompt_text, add_special_tokens=False)]
                    if not token_ids:
                        token_ids = [int(getattr(tokenizer, "bos_token_id", 0) or 0)]
                    generated: list[int] = []
                    preload_seconds = 0.0
                    prompt_embedding_preload_seconds = 0.0
                    prompt_embedding_cache: Any | None = None
                    t034_embedding_table: Any | None = None
                    t034_embedding_preload_seconds = 0.0
                    if dense_cache is not None:
                        dense_started = time.perf_counter()
                        dense_names: list[str] = ["model.norm.weight"]
                        for layer_index in range(requested_layers):
                            prefix = f"model.layers.{layer_index}"
                            dense_names.extend(
                                [
                                    f"{prefix}.input_layernorm.weight",
                                    f"{prefix}.post_attention_layernorm.weight",
                                    f"{prefix}.self_attn.q_norm.weight",
                                    f"{prefix}.self_attn.k_norm.weight",
                                ]
                            )
                        for dense_name in dense_names:
                            dense_cache[dense_name] = _read_tensor(weight_dir, weight_map, dense_name, device=runtime_device)
                        preload_seconds += time.perf_counter() - dense_started
                    if preload_prompt_embeddings:
                        prompt_embedding_started = time.perf_counter()
                        prompt_embedding_cache = _read_rows(
                            weight_dir,
                            weight_map,
                            "model.embed_tokens.weight",
                            [int(token) for token in token_ids],
                            device=runtime_device,
                        )
                        prompt_embedding_preload_seconds = time.perf_counter() - prompt_embedding_started
                        preload_seconds += prompt_embedding_preload_seconds
                    if t034_device_argmax_embedding_splice:
                        t034_embedding_started = time.perf_counter()
                        t034_embedding_table = _read_tensor(
                            weight_dir,
                            weight_map,
                            "model.embed_tokens.weight",
                            device=runtime_device,
                        )
                        t034_embedding_preload_seconds = time.perf_counter() - t034_embedding_started
                        preload_seconds += t034_embedding_preload_seconds
                    if preload_payloads or preload_device_payloads:
                        preload_payload_cache = payload_cache
                        if preload_payloads and payload_cache is None:
                            payload_cache = {}
                            preload_payload_cache = payload_cache
                        preload_seconds = _preload_qwen_payloads(
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            cache_dir=resolved_cache_dir,
                            stats=stats,
                            requested_layers=requested_layers,
                            lm_head_chunk_rows=lm_head_chunk_rows,
                            target_mlp_bits=target_mlp_bits,
                            refresh_cache=refresh_cache,
                            payload_cache=preload_payload_cache,
                            device_payload_cache=device_payloads,
                            device_payload_cache_device=runtime_device if preload_device_payloads else None,
                            device_payload_cache_budget_bytes=device_payload_budget_bytes,
                            binary_cache=binary_cache,
                            trit_packed_codes=trit_packed_codes,
                            trit_code_group_size=trit_code_group_size,
                            device_payload_descriptors=device_payload_descriptors,
                        ) + preload_seconds
                    if fail_closed_device_payloads and not preload_device_payloads:
                        blockers.append("fail_closed_device_payloads_requires_preload_device_payloads")
                    if fail_closed_device_payloads and device_payloads is None:
                        blockers.append("fail_closed_device_payloads_requires_device_payload_cache")
                    pre_hot_load_seconds = stats.load_seconds + stats.binary_load_seconds
                    pre_hot_load_events = stats.tensors_loaded + stats.binary_payloads_loaded + stats.trit_payloads_loaded
                    if runtime_device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    synchronize_kernel = not defer_kernel_synchronization
                    denominator_total, denominator_slack = _t014_a_denominator(config)
                    completed_layers = 0
                    kernel_seconds = 0.0
                    lm_head_certificate: dict[str, Any] | None = None
                    for _ in range(max_new_tokens):
                        if prompt_embedding_cache is not None:
                            if generated:
                                generated_x = _read_rows(
                                    weight_dir,
                                    weight_map,
                                    "model.embed_tokens.weight",
                                    [int(token) for token in generated],
                                    device=runtime_device,
                                )
                                x = torch.cat((prompt_embedding_cache, generated_x), dim=0)
                            else:
                                x = prompt_embedding_cache
                        else:
                            rows = [int(token) for token in token_ids + generated]
                            x = _read_rows(weight_dir, weight_map, "model.embed_tokens.weight", rows, device=runtime_device)
                        completed_layers = 0
                        for layer_index in range(requested_layers):
                            prefix = f"model.layers.{layer_index}"
                            input_norm_name = f"{prefix}.input_layernorm.weight"
                            post_norm_name = f"{prefix}.post_attention_layernorm.weight"
                            q_norm_name = f"{prefix}.self_attn.q_norm.weight"
                            k_norm_name = f"{prefix}.self_attn.k_norm.weight"
                            input_norm = dense_cache[input_norm_name] if dense_cache is not None else _read_tensor(weight_dir, weight_map, input_norm_name, device=runtime_device)
                            post_norm = dense_cache[post_norm_name] if dense_cache is not None else _read_tensor(weight_dir, weight_map, post_norm_name, device=runtime_device)
                            q_norm = dense_cache[q_norm_name] if dense_cache is not None else _read_tensor(weight_dir, weight_map, q_norm_name, device=runtime_device)
                            k_norm = dense_cache[k_norm_name] if dense_cache is not None else _read_tensor(weight_dir, weight_map, k_norm_name, device=runtime_device)
                            x, dt = _attention_forward_kernel(
                                x,
                                model_dir=weight_dir,
                                weight_map=weight_map,
                                cache_dir=resolved_cache_dir,
                                stats=stats,
                                layer_index=layer_index,
                                input_norm=input_norm,
                                q_norm=q_norm,
                                k_norm=k_norm,
                                rms_eps=float(config.as_dict().get("rms_norm_eps", 1e-6)),
                                rope_theta=1_000_000.0,
                                target_mlp_bits=target_mlp_bits,
                                refresh_cache=refresh_cache,
                                payload_cache=payload_cache,
                                device_payload_cache=device_payloads,
                                device_payload_cache_device=runtime_device if device_payloads is not None else None,
                                device_payload_cache_budget_bytes=device_payload_budget_bytes,
                                binary_cache=binary_cache,
                                trit_packed_codes=trit_packed_codes,
                                trit_code_group_size=trit_code_group_size,
                                packed_work_coalescing=packed_work_coalescing,
                                fail_closed_device_payloads=fail_closed_device_payloads,
                                device_payload_descriptors=device_payload_descriptors,
                                synchronize_kernel=synchronize_kernel,
                                packed_linear_block_m=packed_linear_block_m,
                                packed_linear_block_c=packed_linear_block_c,
                                packed_linear_block_r=packed_linear_block_r,
                                native_attention_sdp=native_attention_sdp,
                                fused_qkv_compact_projection_kernel=fused_qkv_compact_projection_kernel,
                            )
                            kernel_seconds += dt
                            x, dt = _mlp_forward_kernel(
                                x,
                                model_dir=weight_dir,
                                weight_map=weight_map,
                                cache_dir=resolved_cache_dir,
                                stats=stats,
                                layer_index=layer_index,
                                post_norm=post_norm,
                                rms_eps=float(config.as_dict().get("rms_norm_eps", 1e-6)),
                                target_mlp_bits=target_mlp_bits,
                                refresh_cache=refresh_cache,
                                payload_cache=payload_cache,
                                device_payload_cache=device_payloads,
                                device_payload_cache_device=runtime_device if device_payloads is not None else None,
                                device_payload_cache_budget_bytes=device_payload_budget_bytes,
                                binary_cache=binary_cache,
                                trit_packed_codes=trit_packed_codes,
                                trit_code_group_size=trit_code_group_size,
                                packed_work_coalescing=packed_work_coalescing,
                                inplace_mlp_intermediate=inplace_mlp_intermediate,
                                fail_closed_device_payloads=fail_closed_device_payloads,
                                device_payload_descriptors=device_payload_descriptors,
                                synchronize_kernel=synchronize_kernel,
                                packed_linear_block_m=packed_linear_block_m,
                                packed_linear_block_c=packed_linear_block_c,
                                packed_linear_block_r=packed_linear_block_r,
                                fused_mlp_gate_up_kernel=fused_mlp_gate_up_kernel,
                            )
                            kernel_seconds += dt
                            completed_layers += 1
                        final_norm = dense_cache["model.norm.weight"] if dense_cache is not None else _read_tensor(weight_dir, weight_map, "model.norm.weight", device=runtime_device)
                        x = _rms_norm(x, final_norm, 1e-6)
                        next_token, dt, lm_head_certificate = _lm_head_argmax_kernel(
                            x[-1],
                            model_dir=weight_dir,
                            weight_map=weight_map,
                            cache_dir=resolved_cache_dir,
                            stats=stats,
                            chunk_rows=lm_head_chunk_rows,
                            target_mlp_bits=target_mlp_bits,
                            refresh_cache=refresh_cache,
                            payload_cache=payload_cache,
                            device_payload_cache=device_payloads,
                            device_payload_cache_device=runtime_device if device_payloads is not None else None,
                            device_payload_cache_budget_bytes=device_payload_budget_bytes,
                            binary_cache=binary_cache,
                            trit_packed_codes=trit_packed_codes,
                            trit_code_group_size=trit_code_group_size,
                            fail_closed_device_payloads=fail_closed_device_payloads,
                            device_payload_descriptors=device_payload_descriptors,
                            synchronize_kernel=synchronize_kernel,
                            gpu_argmax_head=gpu_argmax_head,
                            t034_device_argmax_embedding_splice=t034_device_argmax_embedding_splice,
                            embedding_weight=t034_embedding_table,
                            packed_linear_block_m=packed_linear_block_m,
                            packed_linear_block_c=packed_linear_block_c,
                            packed_linear_block_r=packed_linear_block_r,
                        )
                        kernel_seconds += dt
                        generated.append(next_token)
                    if runtime_device == "cuda" and defer_kernel_synchronization:
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    peak_allocated, peak_reserved = _cuda_memory(runtime_device)
                    process_memory = process_memory_snapshot()
                    generated_text = tokenizer.decode(generated, skip_special_tokens=False)
                    digest = hashlib.sha256(json.dumps({"prompt": prompt_text, "generated": generated}).encode("utf-8")).hexdigest()
                    credit_flow: dict[str, Any] | None = None
                    if device_credit_flow:
                        payload_load_seconds = stats.load_seconds + stats.binary_load_seconds + preload_seconds
                        credit_flow = run_t020_device_credit_flow_gate(
                            baseline_total_seconds=elapsed + preload_seconds,
                            baseline_payload_load_seconds=payload_load_seconds,
                            baseline_kernel_seconds=kernel_seconds,
                            baseline_tokens_per_second=len(generated) / elapsed if elapsed > 0 else 1e-9,
                            device_budget_gib=12.0,
                            baseline_denominator_gib=denominator_total,
                        )
                    receipt = KernelInferenceReceipt(
                        prompt_text=prompt_text,
                        tokenizer_source=tokenizer_source,
                        prompt_token_count=len(token_ids),
                        generated_token_ids=tuple(generated),
                        generated_text=generated_text,
                        requested_layers=requested_layers,
                        completed_layers=completed_layers,
                        device=runtime_device,
                        elapsed_seconds=elapsed,
                        tokens_per_second=len(generated) / elapsed if elapsed > 0 else 0.0,
                        kernel_seconds=kernel_seconds,
                        peak_cuda_allocated_gib=peak_allocated,
                        peak_cuda_reserved_gib=peak_reserved,
                        peak_process_working_set_gib=process_memory.get("peak_working_set_gib"),
                        process_working_set_gib=process_memory.get("working_set_gib"),
                        process_private_usage_gib=process_memory.get("private_usage_gib"),
                        denominator_total_gib=denominator_total,
                        denominator_slack_gib=denominator_slack,
                        cache_stats=stats.as_dict(),
                        output_sha256=digest,
                        resident_payload_cache=resident_payload_cache or preload_payloads,
                        preload_payloads=preload_payloads,
                        preload_seconds=preload_seconds,
                        device_payload_cache=device_payload_cache or preload_device_payloads,
                        device_payload_budget_gib=device_payload_budget_gib,
                        preload_device_payloads=preload_device_payloads,
                        dense_tensor_cache=dense_tensor_cache,
                        trit_packed_codes=trit_packed_codes,
                        device_credit_flow=credit_flow,
                        packed_work_coalescing=coalescing_gate,
                        fail_closed_device_payloads=fail_closed_device_payloads,
                        device_payload_descriptor_table=device_payload_descriptors,
                        hot_payload_load_seconds=(stats.load_seconds + stats.binary_load_seconds) - pre_hot_load_seconds,
                        hot_payload_load_events=(stats.tensors_loaded + stats.binary_payloads_loaded + stats.trit_payloads_loaded)
                        - pre_hot_load_events,
                        defer_kernel_synchronization=defer_kernel_synchronization,
                        gpu_argmax_head=gpu_argmax_head,
                        kernel_timing_mode="deferred_submit" if defer_kernel_synchronization else "synchronized",
                        packed_linear_block_m=packed_linear_block_m,
                        packed_linear_block_c=packed_linear_block_c,
                        packed_linear_block_r=packed_linear_block_r,
                        fused_mlp_gate_up_kernel=fused_mlp_gate_up_kernel,
                        fused_qkv_compact_projection_kernel=fused_qkv_compact_projection_kernel,
                        prompt_embeddings_preloaded=prompt_embedding_cache is not None,
                        prompt_embedding_preload_seconds=prompt_embedding_preload_seconds,
                        native_attention_sdp=native_attention_sdp,
                        trit_code_group_size=trit_code_group_size,
                        t031_lm_head_certificate=lm_head_certificate,
                        t034_device_argmax_embedding_splice={
                            "requested": t034_device_argmax_embedding_splice,
                            "embedding_table_preloaded": t034_embedding_table is not None,
                            "embedding_table_preload_seconds": t034_embedding_preload_seconds,
                            "lm_head_certificate": (lm_head_certificate or {}).get(
                                "t034_device_argmax_embedding_splice"
                            ),
                        },
                    )
        except Exception as exc:  # pragma: no cover - real runtime guard
            blockers.append(f"t017_kernel_qwen_failed:{type(exc).__name__}:{exc}")

    smoke_pass = receipt.passes if receipt is not None and not blockers else False
    if not smoke_pass and not blockers:
        blockers.append("t017_kernel_qwen_failed_threshold")

    return {
        "probe": "t017_kernel_qwen_inference",
        "activation": ACTIVATION,
        "model_id": model_id,
        "model_dir": str(weight_dir) if weight_dir is not None else None,
        "config": config.as_dict(),
        "weight_inventory": inventory.as_dict(),
        "receipt": receipt.as_dict() if receipt is not None else None,
        "missing_tensors": missing,
        "readiness": {
            "smoke_pass": smoke_pass,
            "kernel_qwen_inference_ready": bool(receipt is not None and smoke_pass),
            "full_model_text_inference_ready": bool(receipt is not None and receipt.requested_layers == config.layers and smoke_pass),
            "measured_tokens_per_second": receipt.tokens_per_second if receipt is not None else None,
            "kernel_seconds": receipt.kernel_seconds if receipt is not None else None,
            "config_matches_project_qwen3_32b_shape": config.matches_project_qwen3_32b_shape(),
            "local_weights_present": inventory.local_weights_present,
            "requested_device": device,
            "layer_limit": layer_limit,
            "max_new_tokens": max_new_tokens,
            "refresh_cache": refresh_cache,
            "resident_payload_cache": resident_payload_cache or preload_payloads,
            "preload_payloads": preload_payloads,
            "preload_seconds": receipt.preload_seconds if receipt is not None else None,
            "device_payload_cache": device_payload_cache or preload_device_payloads,
            "device_payload_budget_gib": device_payload_budget_gib,
            "preload_device_payloads": preload_device_payloads,
            "dense_tensor_cache": dense_tensor_cache,
            "binary_cache": binary_cache,
            "trit_packed_codes": trit_packed_codes,
            "trit_code_group_size": trit_code_group_size,
            "device_credit_flow": device_credit_flow,
            "device_credit_flow_gate_ready": bool(
                receipt is not None
                and receipt.device_credit_flow is not None
                and receipt.device_credit_flow["readiness"]["device_credit_flow_gate_ready"]
            ),
            "packed_work_coalescing": packed_work_coalescing,
            "packed_work_coalescing_gate_ready": bool(
                receipt is not None
                and receipt.packed_work_coalescing is not None
                and receipt.packed_work_coalescing["readiness"]["packed_work_coalescing_gate_ready"]
            ),
            "inplace_mlp_intermediate": inplace_mlp_intermediate,
            "fail_closed_device_payloads": fail_closed_device_payloads,
            "device_payload_descriptor_table": bool(
                receipt is not None and receipt.device_payload_descriptor_table is not None
            ),
            "device_payload_descriptor_count": len(receipt.device_payload_descriptor_table or {}) if receipt is not None else 0,
            "hot_payload_load_seconds": receipt.hot_payload_load_seconds if receipt is not None else None,
            "hot_payload_load_events": receipt.hot_payload_load_events if receipt is not None else None,
            "hot_payload_load_free": bool(
                receipt is not None and receipt.hot_payload_load_events == 0 and receipt.hot_payload_load_seconds == 0.0
            ),
            "defer_kernel_synchronization": defer_kernel_synchronization,
            "gpu_argmax_head": gpu_argmax_head,
            "kernel_timing_mode": receipt.kernel_timing_mode if receipt is not None else None,
            "packed_linear_block_m": packed_linear_block_m,
            "packed_linear_block_c": packed_linear_block_c,
            "packed_linear_block_r": packed_linear_block_r,
            "fused_mlp_gate_up_kernel": fused_mlp_gate_up_kernel,
            "fused_qkv_compact_projection_kernel": fused_qkv_compact_projection_kernel,
            "prompt_embeddings_preloaded": receipt.prompt_embeddings_preloaded if receipt is not None else False,
            "prompt_embedding_preload_seconds": receipt.prompt_embedding_preload_seconds if receipt is not None else None,
            "native_attention_sdp": receipt.native_attention_sdp if receipt is not None else False,
            "t034_device_argmax_embedding_splice": t034_device_argmax_embedding_splice,
            "t034_device_argmax_embedding_splice_ready": bool(
                receipt is not None
                and (
                    (receipt.t034_device_argmax_embedding_splice or {}).get("lm_head_certificate") or {}
                ).get("next_embedding_device_closure_proven")
            ),
        },
        "blockers": blockers,
        "next_actions": (
            ["Remove torch.load from the hot path with a resident packed payload arena or mmap format."]
            if smoke_pass and receipt is not None and receipt.requested_layers == config.layers
            else ["Expand the T017 kernel path to the all-layer hot-cache run and compare against T016."]
        )
        if smoke_pass
        else ["Fix the listed T017 Qwen kernel blockers before expanding."],
    }
