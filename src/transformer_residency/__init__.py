"""Transformer residency probes for dense consumer-GPU experiments."""

from .attention import attention_simplex_drift, kv_trust_horizon
from .certificates import gate_cliff_certificate, margin_certificate
from .footprint import TransformerShape, qwen3_32b_shape
from .fragments import FragmentClause, compile_fragment_schedule
from .kv_margin import (
    KVMarginCandidate,
    KVMarginScenario,
    kv_cache_gib_for_context,
    qwen3_32b_kv_bytes_per_token,
    run_t002_kv_margin_sweep,
    t002_candidates,
)
from .policies import (
    AdaptiveMode,
    patch_atom_certificate,
    residual_correlation_alarm,
    select_adaptive_mode,
    support_exit_certificate,
)
from .receipts import TileReceipt
from .residuals import ResidualCollapseReport, residual_collapse_report
from .suites import GateCheck, GateSuiteReport, run_default_gate_suite
from .t003_ram_efficiency import (
    ByteTokSPredictor,
    KiviPlusKVPolicy,
    RepairByteLedger,
    ResidencyReceipt,
    run_t003_receipt_kv_gate,
)
from .t004_real_qwen_bridge import (
    QwenConfigSnapshot,
    SafetensorsInventory,
    inventory_safetensors,
    resolve_qwen_config,
    run_t004_real_qwen_bridge,
)
from .t005_real_weight_sentinel import (
    DEFAULT_SENTINEL_TENSORS,
    TensorSentinel,
    run_t005_real_weight_sentinel,
)
from .t006_real_residency_block_smoke import (
    DEFAULT_BLOCK_TENSORS,
    BlockCodecPolicy,
    BlockSmokeResult,
    run_t006_real_residency_block_smoke,
)
from .t007_real_layer_smoke import (
    LayerSmokeReceipt,
    LayerTensorReceipt,
    run_t007_real_layer_residency_smoke,
)
from .t008_compressed_generation_smoke import GenerationSmokeReceipt, run_t008_compressed_generation_smoke
from .t009_t013_runtime_scaler import ProjectionSmokeReceipt, run_t009_t013_runtime_scaler
from .t014_denominator_tightening import DenominatorCandidate, run_t014_denominator_tightening
from .t015_packed_qwen_inference import StreamedInferenceReceipt, run_t015_packed_qwen_inference
from .t016_persistent_packed_cache import (
    PackedCacheStats,
    PersistentCacheInferenceReceipt,
    run_t016_persistent_packed_cache,
)
from .t017_packed_linear_kernel import (
    KernelInferenceReceipt,
    PackedLinearKernelReceipt,
    run_t017_kernel_qwen_inference,
    run_t017_packed_linear_kernel_gate,
)
from .t018_resident_payload_runtime import run_t018_resident_payload_qwen_inference
from .t019_binary_payload_runtime import run_t019_binary_payload_qwen_inference
from .t020_acceptance_weighted_kv_materialization import run_t020_acceptance_weighted_kv_materialization_gate
from .t020_affine_packed_view import run_t020_affine_packed_view_gate
from .t020_basis_shared_kv_transform import run_t020_basis_shared_kv_transform_gate
from .t020_conversion_time_arena_compiler import run_t020_conversion_time_arena_compiler_gate
from .t020_cumulative_execution_shape import run_t020_cumulative_execution_shape_gate
from .t020_device_credit_flow import run_t020_device_credit_flow_gate
from .t020_expected_accepted_span import run_t020_expected_accepted_span_gate
from .t020_execution_order_slab import layer1_execution_order_objects, run_t020_execution_order_slab_gate
from .t020_immutable_arena_receipt import (
    build_layer1_arena_manifest,
    run_t020_immutable_arena_receipt_gate,
    verify_layer1_arena_manifest,
)
from .t020_packed_work_coalescing import run_t020_packed_work_coalescing_gate
from .t020_prefix_transaction_ledger import run_t020_prefix_transaction_ledger_gate
from .t020_recompute_on_reject import run_t020_recompute_on_reject_gate
from .t020_ring_exact_tail_receipts import run_t020_ring_exact_tail_receipts_gate
from .t021_cumulative_decoder_runtime_path import run_t021_cumulative_decoder_runtime_path_gate
from .t021_page_activation_certificate import run_t021_page_activation_certificate_gate
from .t021_qwen_runtime import run_t021_qwen_runtime
from .t021_runtime_obligation_measure import run_t021_runtime_obligation_measure_gate
from .t021_transaction_receipt_gate import run_t021_transaction_receipt_gate
from .t022_qwen_runtime import run_t022_qwen_runtime
from .t023_qwen_runtime import run_t023_qwen_runtime
from .t024_qwen_runtime import run_t024_qwen_runtime
from .t025_qwen_runtime import run_t025_qwen_runtime
from .t026_qwen_runtime import run_t026_qwen_runtime
from .t027_qwen_runtime import run_t027_qwen_runtime
from .t028_native_generation_speed import run_t028_native_generation_speed
from .t029_speed_gpu_efficiency import run_t029_speed_gpu_efficiency
from .t030_speed_gpu_efficiency import run_t030_speed_gpu_efficiency
from .t031_macro_architecture_stack import run_t031_macro_architecture_stack
from .t034_gpu_real_runtime_gates import run_t034_gpu_real_runtime_gates
from .t035_runtime_guardrails import (
    run_t035_row_successor_gate,
    run_t035_row_successor_runtime_path,
    run_t035_runtime_guardrail_gate,
)
from .t036_block_verifier_gate import run_t036_a05_block_verifier_gate
from .t037_direct_denominator_gate import run_t037_direct_denominator_gate
from .t037_exact_suffix_prefill_gate import run_t037_exact_suffix_prefill_gate
from .t038_kernel_body_roofline_gate import (
    run_t038_kernel_body_roofline_gate,
    run_t038_representative_map_gate,
    run_t038b_coverage_denominator_gate,
)
from .t039_exact_map_body_gate import (
    run_t039_a01_exact_map_object_ledger_gate,
    run_t039_a02_semantic_layout_ledger_gate,
    run_t039b_a01_non_row_exact_object_extraction_ledger,
    run_t039b_a02_gauge_sat_canonical_tail_ledger,
    run_t039_real_map_object_audit,
)

__all__ = [
    "AdaptiveMode",
    "BlockCodecPolicy",
    "BlockSmokeResult",
    "FragmentClause",
    "GateCheck",
    "GateSuiteReport",
    "GenerationSmokeReceipt",
    "KVMarginCandidate",
    "KVMarginScenario",
    "LayerSmokeReceipt",
    "LayerTensorReceipt",
    "ByteTokSPredictor",
    "DenominatorCandidate",
    "KiviPlusKVPolicy",
    "RepairByteLedger",
    "ResidualCollapseReport",
    "ResidencyReceipt",
    "QwenConfigSnapshot",
    "ProjectionSmokeReceipt",
    "PackedCacheStats",
    "PersistentCacheInferenceReceipt",
    "PackedLinearKernelReceipt",
    "KernelInferenceReceipt",
    "SafetensorsInventory",
    "StreamedInferenceReceipt",
    "TensorSentinel",
    "TileReceipt",
    "TransformerShape",
    "DEFAULT_SENTINEL_TENSORS",
    "DEFAULT_BLOCK_TENSORS",
    "attention_simplex_drift",
    "compile_fragment_schedule",
    "gate_cliff_certificate",
    "inventory_safetensors",
    "kv_trust_horizon",
    "kv_cache_gib_for_context",
    "margin_certificate",
    "patch_atom_certificate",
    "qwen3_32b_shape",
    "qwen3_32b_kv_bytes_per_token",
    "resolve_qwen_config",
    "residual_collapse_report",
    "residual_correlation_alarm",
    "run_default_gate_suite",
    "run_t003_receipt_kv_gate",
    "run_t004_real_qwen_bridge",
    "run_t005_real_weight_sentinel",
    "run_t006_real_residency_block_smoke",
    "run_t007_real_layer_residency_smoke",
    "run_t008_compressed_generation_smoke",
    "run_t009_t013_runtime_scaler",
    "run_t014_denominator_tightening",
    "run_t015_packed_qwen_inference",
    "run_t016_persistent_packed_cache",
    "run_t017_packed_linear_kernel_gate",
    "run_t017_kernel_qwen_inference",
    "run_t018_resident_payload_qwen_inference",
    "run_t019_binary_payload_qwen_inference",
    "run_t020_acceptance_weighted_kv_materialization_gate",
    "run_t020_affine_packed_view_gate",
    "run_t020_basis_shared_kv_transform_gate",
    "run_t020_conversion_time_arena_compiler_gate",
    "run_t020_cumulative_execution_shape_gate",
    "run_t020_device_credit_flow_gate",
    "run_t020_expected_accepted_span_gate",
    "run_t020_execution_order_slab_gate",
    "build_layer1_arena_manifest",
    "run_t020_immutable_arena_receipt_gate",
    "run_t020_packed_work_coalescing_gate",
    "run_t020_prefix_transaction_ledger_gate",
    "run_t020_recompute_on_reject_gate",
    "run_t020_ring_exact_tail_receipts_gate",
    "run_t021_cumulative_decoder_runtime_path_gate",
    "run_t021_page_activation_certificate_gate",
    "run_t021_qwen_runtime",
    "run_t021_runtime_obligation_measure_gate",
    "run_t021_transaction_receipt_gate",
    "run_t022_qwen_runtime",
    "run_t023_qwen_runtime",
    "run_t024_qwen_runtime",
    "run_t025_qwen_runtime",
    "run_t026_qwen_runtime",
    "run_t027_qwen_runtime",
    "run_t028_native_generation_speed",
    "run_t029_speed_gpu_efficiency",
    "run_t030_speed_gpu_efficiency",
    "run_t031_macro_architecture_stack",
    "run_t034_gpu_real_runtime_gates",
    "run_t035_row_successor_gate",
    "run_t035_row_successor_runtime_path",
    "run_t035_runtime_guardrail_gate",
    "run_t036_a05_block_verifier_gate",
    "run_t037_direct_denominator_gate",
    "run_t037_exact_suffix_prefill_gate",
    "run_t038_kernel_body_roofline_gate",
    "run_t038_representative_map_gate",
    "run_t038b_coverage_denominator_gate",
    "run_t039_a01_exact_map_object_ledger_gate",
    "run_t039_a02_semantic_layout_ledger_gate",
    "run_t039b_a01_non_row_exact_object_extraction_ledger",
    "run_t039b_a02_gauge_sat_canonical_tail_ledger",
    "run_t039_real_map_object_audit",
    "verify_layer1_arena_manifest",
    "layer1_execution_order_objects",
    "run_t002_kv_margin_sweep",
    "select_adaptive_mode",
    "support_exit_certificate",
    "t002_candidates",
]
