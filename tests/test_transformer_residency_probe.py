import json
import struct
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from src.transformer_residency import (
    TileReceipt,
    KiviPlusKVPolicy,
    RepairByteLedger,
    ResidencyReceipt,
    gate_cliff_certificate,
    inventory_safetensors,
    kv_cache_gib_for_context,
    margin_certificate,
    qwen3_32b_kv_bytes_per_token,
    qwen3_32b_shape,
    resolve_qwen_config,
    residual_collapse_report,
    run_default_gate_suite,
    run_t003_receipt_kv_gate,
    run_t004_real_qwen_bridge,
    run_t005_real_weight_sentinel,
    run_t006_real_residency_block_smoke,
    run_t007_real_layer_residency_smoke,
    run_t008_compressed_generation_smoke,
    run_t009_t013_runtime_scaler,
    run_t014_denominator_tightening,
    run_t015_packed_qwen_inference,
    run_t016_persistent_packed_cache,
    run_t017_kernel_qwen_inference,
    run_t017_packed_linear_kernel_gate,
    run_t018_resident_payload_qwen_inference,
    run_t019_binary_payload_qwen_inference,
    run_t020_acceptance_weighted_kv_materialization_gate,
    run_t020_affine_packed_view_gate,
    run_t020_basis_shared_kv_transform_gate,
    run_t020_conversion_time_arena_compiler_gate,
    run_t020_cumulative_execution_shape_gate,
    run_t020_device_credit_flow_gate,
    run_t020_expected_accepted_span_gate,
    run_t020_execution_order_slab_gate,
    run_t020_immutable_arena_receipt_gate,
    run_t020_packed_work_coalescing_gate,
    run_t020_prefix_transaction_ledger_gate,
    run_t020_recompute_on_reject_gate,
    run_t020_ring_exact_tail_receipts_gate,
    run_t021_cumulative_decoder_runtime_path_gate,
    run_t021_page_activation_certificate_gate,
    run_t021_qwen_runtime,
    run_t021_runtime_obligation_measure_gate,
    run_t021_transaction_receipt_gate,
    run_t022_qwen_runtime,
    run_t023_qwen_runtime,
    run_t024_qwen_runtime,
    run_t025_qwen_runtime,
    run_t026_qwen_runtime,
    run_t027_qwen_runtime,
    run_t028_native_generation_speed,
    run_t029_speed_gpu_efficiency,
    run_t030_speed_gpu_efficiency,
    run_t031_macro_architecture_stack,
    run_t035_row_successor_gate,
    run_t035_row_successor_runtime_path,
    run_t035_runtime_guardrail_gate,
    run_t036_a05_block_verifier_gate,
    run_t037_direct_denominator_gate,
    run_t037_exact_suffix_prefill_gate,
    run_t038_kernel_body_roofline_gate,
    run_t038_representative_map_gate,
    run_t038b_coverage_denominator_gate,
    run_t039_a01_exact_map_object_ledger_gate,
    run_t039_a02_semantic_layout_ledger_gate,
    run_t039b_a01_non_row_exact_object_extraction_ledger,
    run_t039b_a02_gauge_sat_canonical_tail_ledger,
    run_t002_kv_margin_sweep,
    t002_candidates,
)
from src.transformer_residency.t039_exact_map_body_gate import _exact_row_quotient_metrics


class TransformerResidencyProbeTests(unittest.TestCase):
    def test_tile_receipt_counts_hidden_costs(self):
        receipt = TileReceipt(
            dense_elements=1024,
            payload_bits_per_element=2.5,
            metadata_bytes=64,
            scale_bytes=64,
            correction_bytes=128,
            scratch_bytes=0,
        )

        self.assertLess(receipt.execution_bytes, receipt.dense_bytes)
        self.assertTrue(receipt.passes(0.10))
        self.assertGreater(receipt.effective_bits_per_element, 2.5)

    def test_residual_collapse_detects_down_invisible_noise(self):
        rng = np.random.default_rng(123)
        samples = 64
        intermediate = 32
        hidden = 8
        down = np.zeros((intermediate, hidden))
        down[:hidden, :] = np.eye(hidden) * 0.1
        residual = rng.normal(size=(samples, intermediate))
        residual[:, :hidden] *= 0.05

        report = residual_collapse_report(residual, down)

        self.assertTrue(report.passes_energy_gate)
        self.assertLess(report.energy_ratio, 0.35)

    def test_margin_certificate_rejects_close_logits(self):
        close = margin_certificate(np.array([1.0, 0.95, 0.1]), error_radius=0.04)
        far = margin_certificate(np.array([1.0, 0.50, 0.1]), error_radius=0.04)

        self.assertFalse(close.accepted)
        self.assertTrue(far.accepted)

    def test_gate_cliff_certificate_rejects_near_boundary_values(self):
        gate_values = np.array([-0.5, -0.2, 0.01, 0.4])

        cert = gate_cliff_certificate(
            gate_values,
            residual_bound=0.1,
            margin=0.05,
            min_accepted_fraction=0.9,
        )

        self.assertFalse(cert.accepted)
        self.assertLess(cert.accepted_fraction, 0.9)

    def test_default_gate_suite_matches_positive_and_negative_controls(self):
        suite = run_default_gate_suite(seed=777)

        self.assertTrue(suite.matched)
        self.assertEqual(len(suite.unexpected), 0)
        observed = {check.name: check.observed_pass for check in suite.checks}
        self.assertTrue(observed["residual_collapse_positive_control"])
        self.assertFalse(observed["residual_collapse_negative_control"])
        self.assertTrue(observed["tile_receipt_positive_control"])
        self.assertFalse(observed["tile_receipt_negative_control"])
        self.assertTrue(observed["logit_margin_positive_control"])
        self.assertFalse(observed["logit_margin_negative_control"])
        self.assertTrue(observed["mlp_gate_cliff_positive_control"])
        self.assertFalse(observed["mlp_gate_cliff_negative_control"])
        self.assertTrue(observed["attention_simplex_positive_control"])
        self.assertFalse(observed["attention_simplex_negative_control"])
        self.assertTrue(observed["support_exit_positive_control"])
        self.assertFalse(observed["support_exit_negative_control"])
        self.assertTrue(observed["adaptive_selector_positive_control"])
        self.assertFalse(observed["adaptive_selector_negative_control"])
        self.assertTrue(observed["patch_atom_positive_control"])
        self.assertFalse(observed["patch_atom_negative_control"])
        self.assertTrue(observed["residual_correlation_positive_control"])
        self.assertFalse(observed["residual_correlation_negative_control"])
        self.assertTrue(observed["kv_trust_horizon_positive_control"])
        self.assertFalse(observed["kv_trust_horizon_negative_control"])
        self.assertTrue(observed["fragment_clause_positive_control"])
        self.assertFalse(observed["fragment_clause_negative_control"])

    def test_qwen3_32b_projection_tracks_effective_bits(self):
        shape = qwen3_32b_shape()
        q4 = shape.residency_projection(candidate_bits=4.0, overhead_gib=1.0)
        q25 = shape.residency_projection(candidate_bits=2.5, overhead_gib=1.0)
        stack = shape.component_residency_projection(
            mlp_bits=2.65,
            attention_bits=3.50,
            embedding_bits=4.0,
            output_head_bits=4.0,
            overhead_gib=1.0,
        )

        self.assertGreater(shape.estimated_weight_elements, 30_000_000_000)
        self.assertGreater(shape.estimated_mlp_fraction, 0.75)
        self.assertGreater(q4["candidate_estimated_gib"], 12.0)
        self.assertLess(q25["candidate_estimated_gib"], q4["candidate_estimated_gib"])
        self.assertTrue(q25["candidate_fits_12gib"])
        self.assertLess(stack["candidate_estimated_gib"], q4["candidate_estimated_gib"])
        self.assertTrue(stack["candidate_fits_12gib"])

    def test_synthetic_decode_throughput_tracks_weight_payload(self):
        shape = qwen3_32b_shape()
        throughput = shape.synthetic_decode_throughput_projection(
            uniform_bits=3.0,
            mlp_bits=2.65,
            attention_bits=3.50,
            embedding_bits=4.0,
            output_head_bits=4.0,
            bandwidth_gib_per_second=500.0,
        )
        double_bandwidth = shape.synthetic_decode_throughput_projection(
            uniform_bits=3.0,
            mlp_bits=2.65,
            attention_bits=3.50,
            embedding_bits=4.0,
            output_head_bits=4.0,
            bandwidth_gib_per_second=1000.0,
        )

        self.assertLess(throughput["component_weight_payload_gib"], throughput["uniform_weight_payload_gib"])
        self.assertLess(throughput["uniform_weight_payload_gib"], throughput["q4_weight_payload_gib"])
        self.assertGreater(throughput["component_speedup_vs_q4"], 1.10)
        self.assertGreater(throughput["component_speedup_vs_uniform"], 1.0)
        self.assertAlmostEqual(
            double_bandwidth["q4_tokens_per_second"],
            throughput["q4_tokens_per_second"] * 2.0,
        )

    def test_t002_kv_denominator_matches_qwen3_shape(self):
        shape = qwen3_32b_shape()

        self.assertEqual(qwen3_32b_kv_bytes_per_token(shape), 262_144)
        self.assertAlmostEqual(kv_cache_gib_for_context(4096, shape=shape), 1.0)
        self.assertAlmostEqual(kv_cache_gib_for_context(8192, shape=shape), 2.0)
        self.assertAlmostEqual(kv_cache_gib_for_context(16384, shape=shape), 4.0)
        self.assertAlmostEqual(kv_cache_gib_for_context(32768, shape=shape), 8.0)

    def test_t031_macro_architecture_stack_accounts_all_survivors(self):
        result = run_t031_macro_architecture_stack(execute_runtime=False)
        candidate = result["candidate"]
        speed_projection = candidate["speed_projection"]
        memory_projection = candidate["memory_projection"]
        receipts = candidate["algorithmic_receipts"]

        self.assertEqual(result["activation"], "T031_MACRO_ARCHITECTURE_STACK_ACTIVE")
        self.assertTrue(result["readiness"]["t031_macro_architecture_stack_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t031_all_survivors_accounted"])
        self.assertTrue(result["readiness"]["t031_all_speed_survivors_accounted"])
        self.assertTrue(result["readiness"]["t031_all_efficiency_survivors_accounted"])
        self.assertTrue(result["readiness"]["t031_all_algorithmic_receipts_ready"])
        self.assertTrue(result["readiness"]["t031_no_quality_tradeoff"])
        self.assertTrue(result["readiness"]["t031_gpu_bound_efficiency_only"])
        self.assertTrue(result["readiness"]["t031_overlap_deduplicated_not_naively_multiplied"])
        self.assertEqual(candidate["mechanism_count"], 15)
        self.assertEqual(candidate["speed_mechanism_count"], 7)
        self.assertEqual(candidate["efficiency_mechanism_count"], 8)
        self.assertEqual(len(receipts), 15)
        self.assertTrue(all(receipt["ready"] for receipt in receipts.values()))
        self.assertEqual(receipts["T031-S02"]["dense_result"], receipts["T031-S02"]["fabric_result"])
        self.assertGreater(receipts["T031-S07"]["mean_decoded_length"], 1.0)
        self.assertGreater(speed_projection["projected_tokens_per_second"], result["baseline"]["speed_tokens_per_second"])
        self.assertLess(memory_projection["projected_peak_cuda_gib"], result["baseline"]["speed_peak_cuda_gib"])
        self.assertFalse(candidate["production_tok_s_claim_allowed"])
        self.assertFalse(candidate["measured_runtime_claim"])
        self.assertFalse(result["readiness"]["t031_real_runtime_executed"])

    def test_t035_runtime_guardrail_gate_prices_full64_scope(self):
        result = run_t035_runtime_guardrail_gate()
        self.assertEqual(result["activation"], "T035_RUNTIME_GUARDRAIL_GATE_ACTIVE")
        self.assertTrue(result["readiness"]["t035_runtime_guardrail_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t035_one_layer_not_used_as_full64_baseline"])
        self.assertFalse(result["readiness"]["t035_speed_claim_allowed"])
        self.assertIn("guardrail_is_not_a_standalone_speed_claim", result["blockers"])
        self.assertEqual(
            result["baseline_table"]["one_layer_diagnostic"]["claim_boundary"],
            "diagnostic_only_not_full64_speed_baseline",
        )
        self.assertLessEqual(
            result["scratch_contract"]["candidate_peak_cuda_gib"],
            result["scratch_contract"]["peak_cap_gib"],
        )
        self.assertEqual(result["scratch_contract"]["hidden_dense_live_gib"], 0.0)

    def test_t035_runtime_guardrail_rejects_hidden_dense_shadow(self):
        result = run_t035_runtime_guardrail_gate(hidden_dense_gib=0.5)
        self.assertFalse(result["readiness"]["t035_runtime_guardrail_gate_ready"])
        self.assertIn("hidden_dense_shadow_live", result["blockers"])

    def test_t035_row_successor_gate_preserves_exact_visible_boundary(self):
        result = run_t035_row_successor_gate(
            device="cpu",
            require_cuda=False,
            vocab_rows=256,
            hidden_size=64,
            chunk_rows=32,
        )
        self.assertEqual(result["activation"], "T035_ROW_SUCCESSOR_GATE_ACTIVE")
        self.assertTrue(result["readiness"]["t035_row_successor_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t035_exact_token_match"])
        self.assertTrue(result["readiness"]["t035_exact_embedding_match"])
        self.assertTrue(result["readiness"]["t035_next_embedding_ready_before_host_token"])
        self.assertTrue(result["readiness"]["t035_no_full_logits_in_candidate"])
        self.assertFalse(result["readiness"]["t035_speed_claim_allowed"])

    def test_t035_row_successor_runtime_path_requires_real_splice_contract(self):
        def fake_runner(**kwargs):
            self.assertTrue(kwargs["gpu_argmax_head"])
            self.assertTrue(kwargs["t034_device_argmax_embedding_splice"])
            self.assertTrue(kwargs["device_payload_cache"])
            self.assertTrue(kwargs["preload_device_payloads"])
            self.assertFalse(kwargs["dense_tensor_cache"])
            return {
                "readiness": {"smoke_pass": True},
                "blockers": [],
                "receipt": {
                    "requested_layers": 1,
                    "completed_layers": 1,
                    "device": "cuda",
                    "tokens_per_second": 3.2,
                    "peak_cuda_allocated_gib": 0.6,
                    "peak_cuda_reserved_gib": 0.7,
                    "device_payload_cache": True,
                    "hot_payload_load_events": 0,
                    "hot_payload_load_seconds": 0.0,
                    "cache_stats": {"device_resident_bytes": int(0.52 * (1024**3))},
                    "t034_device_argmax_embedding_splice": {
                        "lm_head_certificate": {
                            "active": True,
                            "next_embedding_device_closure_proven": True,
                            "host_token_id_dependency_before_next_embedding": False,
                            "host_token_id_read_after_device_embedding": True,
                            "full_logits_materialized": False,
                            "full_logits_returned_to_host": False,
                        }
                    },
                },
            }

        result = run_t035_row_successor_runtime_path(runtime_runner=fake_runner)
        self.assertEqual(result["activation"], "T035_ROW_SUCCESSOR_RUNTIME_PATH_ACTIVE")
        self.assertTrue(result["readiness"]["t035_row_successor_runtime_path_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t035_gpu_bound_device_payload_path"])
        self.assertTrue(result["readiness"]["t035_exact_device_splice_closed"])
        self.assertTrue(result["readiness"]["t035_no_full_logits_materialized_or_returned"])
        self.assertFalse(result["readiness"]["t035_production_full64_speed_claim_allowed"])
        self.assertTrue(result["comparison"]["one_layer_speed_claim_allowed"])

    def test_t036_block_verifier_gate_rejects_serial_full64_replay(self):
        result = run_t036_a05_block_verifier_gate()
        self.assertEqual(result["activation"], "T036_A05_BLOCK_VERIFIER_GATE_ACTIVE")
        self.assertFalse(result["readiness"]["t036_a05_block_verifier_gate_ready"])
        self.assertFalse(result["readiness"]["t036_current_executable_path_crosses_15_tok_s"])
        self.assertTrue(result["readiness"]["t036_projection_has_target_passing_condition"])
        self.assertIn("verifier_is_serial_full64_replay", result["blockers"])
        self.assertIn("block_verifier_does_not_reach_15_tok_s", result["blockers"])
        self.assertLess(result["best_conditional_projection_row"]["seconds_per_accepted_token"], 1.0 / 15.0)
        self.assertGreater(result["best_current_row"]["seconds_per_accepted_token"], 1.0 / 15.0)

    def test_t036_block_verifier_gate_rejects_hidden_memory_and_broken_exactness(self):
        result = run_t036_a05_block_verifier_gate(
            verifier_mode="projected_sublinear",
            dense_sidecar_gib=0.5,
            exact_state_equality=False,
        )
        self.assertFalse(result["readiness"]["t036_a05_block_verifier_gate_ready"])
        self.assertIn("hidden_dense_sidecar_live", result["blockers"])
        self.assertIn("exact_state_equality_not_proven", result["blockers"])

    def test_t037_exact_suffix_prefill_gate_validates_toy_exactness_but_requires_full64_timing(self):
        result = run_t037_exact_suffix_prefill_gate(block_sizes=(16,))
        self.assertEqual(result["activation"], "T037_EXACT_SUFFIX_PREFILL_GATE_ACTIVE")
        self.assertTrue(result["readiness"]["t037_suffix_prefill_exactness_ready"], result["blockers"])
        self.assertFalse(result["readiness"]["t037_full64_timing_measured"])
        self.assertFalse(result["readiness"]["t037_exact_suffix_prefill_gate_ready"])
        self.assertFalse(result["readiness"]["t037_production_implementation_allowed"])
        self.assertIn("full64_suffix_prefill_timing_missing", result["blockers"])
        self.assertIn("suffix_prefill_pricing_does_not_reach_15_tok_s", result["blockers"])

    def test_t037_exact_suffix_prefill_gate_accepts_measured_full64_pricing_without_sidecar(self):
        result = run_t037_exact_suffix_prefill_gate(
            block_sizes=(64,),
            full64_suffix_prefill_seconds=2.1550956000573933,
            timing_source="full64_measured",
        )
        self.assertTrue(result["readiness"]["t037_exact_suffix_prefill_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t037_pricing_crosses_15_tok_s"])
        self.assertTrue(result["readiness"]["t037_no_hidden_dense_sidecar"])
        self.assertGreaterEqual(result["best_pricing_row"]["tokens_per_second"], 15.0)

    def test_t037_exact_suffix_prefill_gate_rejects_hidden_dense_sidecar(self):
        result = run_t037_exact_suffix_prefill_gate(
            block_sizes=(64,),
            full64_suffix_prefill_seconds=2.1550956000573933,
            dense_sidecar_gib=0.5,
            timing_source="full64_measured",
        )
        self.assertFalse(result["readiness"]["t037_exact_suffix_prefill_gate_ready"])
        self.assertIn("hidden_dense_sidecar_live", result["blockers"])

    def test_t037_direct_denominator_gate_rejects_current_r452_denominator(self):
        result = run_t037_direct_denominator_gate()
        self.assertEqual(result["activation"], "T037_DIRECT_DENOMINATOR_GATE_ACTIVE")
        self.assertFalse(result["readiness"]["t037_direct_denominator_gate_ready"])
        self.assertFalse(result["readiness"]["t037_control_removal_alone_reaches_15_tok_s"])
        self.assertIn("direct_denominator_does_not_reach_15_tok_s", result["blockers"])
        self.assertIn("control_removal_alone_cannot_reach_target", result["blockers"])
        self.assertGreater(result["best_current_row"]["total_seconds_per_token"], 1.0 / 15.0)
        self.assertGreater(result["control_only_row"]["total_seconds_per_token"], 1.0 / 15.0)

    def test_t037_direct_denominator_gate_accepts_exact_target_edge_only(self):
        result = run_t037_direct_denominator_gate(measured_direct_seconds_per_token=1.0 / 15.0)
        self.assertTrue(result["readiness"]["t037_direct_denominator_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t037_direct_exact_vm_reaches_15_tok_s"])

    def test_t037_direct_denominator_gate_rejects_hidden_sidecar_even_when_fast(self):
        result = run_t037_direct_denominator_gate(
            measured_direct_seconds_per_token=1.0 / 15.0,
            hidden_sidecar_gib=0.125,
        )
        self.assertFalse(result["readiness"]["t037_direct_denominator_gate_ready"])
        self.assertIn("hidden_dense_or_host_sidecar_live", result["blockers"])

    def test_t038_kernel_body_roofline_gate_survives_weight_once_path(self):
        result = run_t038_kernel_body_roofline_gate()
        self.assertEqual(result["activation"], "T038_KERNEL_BODY_ROOFLINE_GATE_ACTIVE")
        self.assertTrue(result["readiness"]["t038_kernel_body_roofline_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t038_hot_byte_roofline_survives"])
        self.assertTrue(result["readiness"]["t038_resident_format_covers_body"])
        self.assertTrue(result["readiness"]["t038_layout_replaces_storage"])
        self.assertFalse(result["readiness"]["t038_production_implementation_allowed"])
        self.assertLess(result["scenarios"][-1]["total_seconds_per_token"], 1.0 / 15.0)

    def test_t038_kernel_body_roofline_gate_rejects_too_many_hot_reads(self):
        result = run_t038_kernel_body_roofline_gate(hot_read_multiplier=8.0)
        self.assertFalse(result["readiness"]["t038_kernel_body_roofline_gate_ready"])
        self.assertIn("hot_byte_roofline_does_not_reach_15_tok_s", result["blockers"])
        self.assertIn("hot_read_multiplier_exceeds_target_budget", result["blockers"])

    def test_t038_kernel_body_roofline_gate_rejects_sidecar_and_noncovering_format(self):
        result = run_t038_kernel_body_roofline_gate(
            compact_affine_map_fraction=0.25,
            sign_plane_map_fraction=0.0,
            dense_map_fraction=0.75,
            decoded_dense_sidecar_gib=0.5,
        )
        self.assertFalse(result["readiness"]["t038_kernel_body_roofline_gate_ready"])
        self.assertIn("hidden_or_decoded_dense_sidecar_live", result["blockers"])
        self.assertIn("replacement_layout_does_not_replace_resident_storage", result["blockers"])
        self.assertIn("resident_format_does_not_cover_enough_body_maps", result["blockers"])

    def test_t038_representative_map_gate_rejects_current_t017_speedup(self):
        result = run_t038_representative_map_gate()
        self.assertEqual(result["activation"], "T038_REPRESENTATIVE_MAP_GATE_ACTIVE")
        self.assertFalse(result["readiness"]["t038_representative_map_gate_ready"])
        self.assertFalse(result["readiness"]["t038_representative_map_speedup_sufficient"])
        self.assertIn("representative_map_speedup_below_required_7_5738x", result["blockers"])
        self.assertLess(result["representative_map"]["projected_tokens_per_second_if_control_zero"], 15.0)

    def test_t038_representative_map_gate_accepts_only_exact_sufficient_map(self):
        result = run_t038_representative_map_gate(representative_speedup_vs_current=8.0)
        self.assertTrue(result["readiness"]["t038_representative_map_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t038_representative_map_speedup_sufficient"])

    def test_t038_representative_map_gate_rejects_fast_inexact_map(self):
        result = run_t038_representative_map_gate(
            representative_speedup_vs_current=8.0,
            exact_output_equivalence=False,
        )
        self.assertFalse(result["readiness"]["t038_representative_map_gate_ready"])
        self.assertIn("representative_map_exact_output_not_proven", result["blockers"])

    def test_t038b_coverage_denominator_gate_rejects_unaccelerated_leftover(self):
        result = run_t038b_coverage_denominator_gate()
        self.assertEqual(result["activation"], "T038B_COVERAGE_DENOMINATOR_GATE_ACTIVE")
        self.assertFalse(result["readiness"]["t038b_coverage_denominator_gate_ready"])
        self.assertTrue(result["readiness"]["t038b_accelerated_body_fraction_sufficient"])
        self.assertFalse(result["readiness"]["t038b_composite_speedup_sufficient"])
        self.assertIn("composite_body_speedup_below_required_7_5738x", result["blockers"])
        self.assertLess(result["coverage_model"]["projected_tokens_per_second_if_control_zero"], 15.0)

    def test_t038b_coverage_denominator_gate_accepts_priced_composite_body(self):
        result = run_t038b_coverage_denominator_gate(
            component_names=("atom_program_body", "tiled_contract_body"),
            component_fractions=(0.90, 0.10),
            component_speedups=(10.0, 8.0),
        )
        self.assertTrue(result["readiness"]["t038b_coverage_denominator_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t038b_composite_speedup_sufficient"])
        self.assertGreaterEqual(result["coverage_model"]["projected_tokens_per_second_if_control_zero"], 15.0)
        self.assertFalse(result["readiness"]["t038b_production_implementation_allowed"])

    def test_t038b_coverage_denominator_gate_rejects_sidecar_and_residual(self):
        result = run_t038b_coverage_denominator_gate(
            component_fractions=(1.0,),
            component_speedups=(10.0,),
            hidden_sidecar_gib=0.25,
            residual_fraction=0.01,
            exact_arithmetic=False,
        )
        self.assertFalse(result["readiness"]["t038b_coverage_denominator_gate_ready"])
        self.assertIn("hidden_or_decoded_dense_sidecar_live", result["blockers"])
        self.assertIn("exact_map_residual_is_nonzero", result["blockers"])
        self.assertIn("exact_arithmetic_not_proven", result["blockers"])

    def test_t039_a01_exact_map_object_ledger_gate_rejects_missing_evidence(self):
        result = run_t039_a01_exact_map_object_ledger_gate()
        self.assertEqual(result["activation"], "T039_A01_EXACT_MAP_OBJECT_LEDGER_GATE_ACTIVE")
        self.assertFalse(result["readiness"]["t039_a01_exact_map_object_ledger_gate_ready"])
        self.assertIn("composite_body_speedup_below_required_7_5738x", result["blockers"])
        self.assertIn("exact_map_object_equivalence_not_proven", result["blockers"])
        self.assertIn("no_representative_real_maps_audited", result["blockers"])

    def test_t039_a01_exact_map_object_ledger_gate_accepts_fully_priced_exact_object(self):
        result = run_t039_a01_exact_map_object_ledger_gate(
            component_names=("tile_sum_program_body",),
            component_fractions=(1.0,),
            component_speedups=(8.0,),
            object_equivalence_proven=True,
            representative_real_maps_audited=4,
            layout_replaces_storage=True,
        )
        self.assertTrue(result["readiness"]["t039_a01_exact_map_object_ledger_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t039_a01_composite_speedup_sufficient"])
        self.assertFalse(result["readiness"]["t039_production_implementation_allowed"])

    def test_t039_a01_exact_map_object_ledger_gate_rejects_sidecar_and_residual(self):
        result = run_t039_a01_exact_map_object_ledger_gate(
            component_fractions=(1.0,),
            component_speedups=(8.0,),
            object_equivalence_proven=True,
            representative_real_maps_audited=1,
            residual_fraction=0.01,
            hidden_sidecar_gib=0.25,
            layout_replaces_storage=True,
        )
        self.assertFalse(result["readiness"]["t039_a01_exact_map_object_ledger_gate_ready"])
        self.assertIn("exact_map_residual_is_nonzero", result["blockers"])
        self.assertIn("hidden_or_decoded_dense_sidecar_live", result["blockers"])

    def test_t039_a02_semantic_layout_ledger_gate_rejects_missing_contracts(self):
        result = run_t039_a02_semantic_layout_ledger_gate()
        self.assertEqual(result["activation"], "T039_A02_SEMANTIC_LAYOUT_LEDGER_GATE_ACTIVE")
        self.assertFalse(result["readiness"]["t039_a02_semantic_layout_ledger_gate_ready"])
        self.assertIn("composite_body_speedup_below_required_7_5738x", result["blockers"])
        self.assertIn("exact_gauge_contracts_not_proven", result["blockers"])
        self.assertIn("semantic_boundaries_not_priced", result["blockers"])
        self.assertIn("forced_canonical_fraction_exceeds_leftover_bound", result["blockers"])

    def test_t039_a02_semantic_layout_ledger_gate_accepts_priced_contract(self):
        result = run_t039_a02_semantic_layout_ledger_gate(
            component_names=("private_tiled_body", "priced_commit_tail"),
            component_fractions=(0.90, 0.10),
            component_speedups=(10.0, 8.0),
            gauge_contracts_proven=True,
            semantic_boundaries_priced=True,
            forced_canonical_fraction=0.05,
            layout_replaces_storage=True,
        )
        self.assertTrue(result["readiness"]["t039_a02_semantic_layout_ledger_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t039_a02_composite_speedup_sufficient"])
        self.assertFalse(result["readiness"]["t039_production_implementation_allowed"])

    def test_t039_a02_semantic_layout_ledger_gate_rejects_duplicate_layout_sidecar(self):
        result = run_t039_a02_semantic_layout_ledger_gate(
            component_fractions=(1.0,),
            component_speedups=(8.0,),
            gauge_contracts_proven=True,
            semantic_boundaries_priced=True,
            forced_canonical_fraction=0.0,
            hidden_sidecar_gib=0.25,
            layout_replaces_storage=False,
        )
        self.assertFalse(result["readiness"]["t039_a02_semantic_layout_ledger_gate_ready"])
        self.assertIn("hidden_or_decoded_dense_sidecar_live", result["blockers"])
        self.assertIn("semantic_layout_does_not_replace_storage", result["blockers"])

    def test_t039_exact_row_quotient_metrics_reconstructs_duplicate_rows(self):
        import torch

        block = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, -1.0],
            ]
        )
        metrics = _exact_row_quotient_metrics(block)
        self.assertTrue(metrics["exact_reconstruction"])
        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["unique_rows"], 2)
        self.assertEqual(metrics["row_quotient_speedup"], 2.0)

    def test_t039b_non_row_exact_object_ledger_rejects_full_rank_synthetic_block(self):
        import torch

        generator = torch.Generator().manual_seed(3901)
        block = torch.randn((16, 16), generator=generator)
        result = run_t039b_a01_non_row_exact_object_extraction_ledger(
            synthetic_blocks=(("synthetic.full_rank", block),)
        )
        self.assertEqual(result["activation"], "T039B_A01_NON_ROW_EXACT_OBJECT_EXTRACTION_LEDGER_ACTIVE")
        self.assertFalse(result["readiness"]["t039b_a01_non_row_exact_object_extraction_ledger_ready"])
        self.assertIn("ledger:composite_body_speedup_below_required_7_5738x", result["blockers"])
        self.assertFalse(result["ledger"]["readiness"]["t039_production_implementation_allowed"])

    def test_t039b_non_row_exact_object_ledger_accepts_exact_low_rank_synthetic_block(self):
        import torch

        row = torch.arange(1, 65, dtype=torch.float32).reshape(1, 64)
        coeff = torch.arange(1, 65, dtype=torch.float32).reshape(64, 1)
        block = coeff @ row
        result = run_t039b_a01_non_row_exact_object_extraction_ledger(
            synthetic_blocks=(("synthetic.rank1", block),)
        )
        self.assertTrue(
            result["readiness"]["t039b_a01_non_row_exact_object_extraction_ledger_ready"],
            result["blockers"],
        )
        self.assertGreaterEqual(result["readiness"]["composite_speedup_vs_current_body"], 7.573753504548222)
        self.assertFalse(result["ledger"]["readiness"]["t039_production_implementation_allowed"])

    def test_t039b_a02_gauge_sat_ledger_rejects_default_canonical_tail(self):
        result = run_t039b_a02_gauge_sat_canonical_tail_ledger()
        self.assertEqual(result["activation"], "T039B_A02_GAUGE_SAT_CANONICAL_TAIL_LEDGER_ACTIVE")
        self.assertFalse(result["readiness"]["t039b_a02_gauge_sat_canonical_tail_ledger_ready"])
        self.assertFalse(result["readiness"]["forced_canonical_tail_below_bound"])
        self.assertIn("ledger:forced_canonical_fraction_exceeds_leftover_bound", result["blockers"])

    def test_t039b_a02_gauge_sat_ledger_accepts_private_regions(self):
        regions = (
            {
                "name": "private_body",
                "body_fraction": 0.90,
                "private_speedup": 40.0,
                "exact_gauge_relation": True,
                "semantic_boundary_priced": True,
                "forced_canonical": False,
                "layout_replaces_storage": True,
            },
            {
                "name": "priced_canonical_tail",
                "body_fraction": 0.10,
                "private_speedup": 1.0,
                "exact_gauge_relation": True,
                "semantic_boundary_priced": True,
                "forced_canonical": True,
                "layout_replaces_storage": True,
            },
        )
        result = run_t039b_a02_gauge_sat_canonical_tail_ledger(regions=regions)
        self.assertTrue(
            result["readiness"]["t039b_a02_gauge_sat_canonical_tail_ledger_ready"],
            result["blockers"],
        )
        self.assertGreaterEqual(result["readiness"]["composite_speedup_vs_current_body"], 7.573753504548222)
        self.assertFalse(result["ledger"]["readiness"]["t039_production_implementation_allowed"])

    def test_t002_candidate_inventory_meets_extraordinary_round_floor(self):
        candidates = t002_candidates()
        strong = [candidate for candidate in candidates if candidate.is_strong]
        near_no_tradeoff_strong = [
            candidate for candidate in candidates if candidate.is_strong and candidate.near_no_tradeoff
        ]

        self.assertEqual(len(candidates), 32)
        self.assertGreaterEqual(len(strong), 20)
        self.assertGreaterEqual(len(near_no_tradeoff_strong), 20)

    def test_t002_kv_margin_sweep_compares_before_after(self):
        result = run_t002_kv_margin_sweep()
        scenarios = {scenario["name"]: scenario for scenario in result["scenarios"]}

        self.assertTrue(result["overall"]["pass"])
        self.assertLess(scenarios["before_t001_component_fp16_kv_4k"]["slack_gib"], 0.05)
        self.assertGreaterEqual(
            scenarios["minimal_static_arena_light_tighten_int8_kv_8k"]["slack_gib"],
            0.35,
        )
        self.assertGreaterEqual(scenarios["strong_weight_tighten_tiered_int4_kv_16k"]["slack_gib"], 0.75)
        self.assertGreaterEqual(scenarios["aggressive_quotient_int2_kv_32k"]["slack_gib"], 0.60)
        self.assertGreaterEqual(
            scenarios["tok_s_push_exact_self_spec_int4_16k"]["speedup_vs_baseline"],
            1.20,
        )
        self.assertEqual(result["activation"], "T002_KV_MARGIN_SWEEP_ACTIVE")

    def test_t003_repair_byte_ledger_charges_hidden_costs(self):
        passing = RepairByteLedger(gross_saved_gib=0.74, repair_atom_gib=0.17, metadata_gib=0.07, scratch_gib=0.03)
        failing = RepairByteLedger(gross_saved_gib=0.68, repair_atom_gib=0.30, metadata_gib=0.08, scratch_gib=0.04)

        self.assertTrue(passing.passes)
        self.assertGreaterEqual(passing.net_saved_gib, 0.25)
        self.assertFalse(failing.passes)
        self.assertGreater(failing.repair_fraction_of_gross_saved, 0.50)

    def test_t003_kivi_plus_kv_policy_requires_residual_and_byte_gates(self):
        shape = qwen3_32b_shape()
        passing = KiviPlusKVPolicy(
            context_tokens=16384,
            exact_recent_tokens=1024,
            old_key_bits=2.0,
            old_value_bits=3.0,
            metadata_overhead_gib=0.03,
            key_logit_error=0.07,
            value_output_error=0.05,
            topk_margin=0.24,
        )
        unsafe_residual = KiviPlusKVPolicy(
            context_tokens=16384,
            exact_recent_tokens=1024,
            old_key_bits=2.0,
            old_value_bits=3.0,
            metadata_overhead_gib=0.03,
            key_logit_error=0.12,
            value_output_error=0.09,
            topk_margin=0.24,
        )
        too_large = KiviPlusKVPolicy(
            context_tokens=16384,
            exact_recent_tokens=4096,
            old_key_bits=4.0,
            old_value_bits=4.0,
            metadata_overhead_gib=0.03,
            key_logit_error=0.04,
            value_output_error=0.04,
            topk_margin=0.24,
        )

        self.assertLess(passing.kv_gib(shape), 0.90)
        self.assertTrue(passing.as_dict(shape)["passes"])
        self.assertFalse(unsafe_residual.as_dict(shape)["passes"])
        self.assertFalse(too_large.as_dict(shape)["passes"])

    def test_t003_residency_receipt_hashes_and_counts_safety(self):
        receipt = ResidencyReceipt(
            weight_runtime_gib=9.1,
            kv_gib=0.85,
            repair_charged_gib=0.0,
            page_table_gib=0.04,
            scratch_gib=0.05,
            safety_gib=0.25,
        )
        same = ResidencyReceipt(
            weight_runtime_gib=9.1,
            kv_gib=0.85,
            repair_charged_gib=0.0,
            page_table_gib=0.04,
            scratch_gib=0.05,
            safety_gib=0.25,
        )

        self.assertEqual(receipt.audit_hash, same.audit_hash)
        self.assertTrue(receipt.fits_with_safety)
        self.assertAlmostEqual(receipt.total_with_safety_gib, receipt.total_without_safety_gib + 0.25)

    def test_t003_receipt_kv_gate_improves_margin_and_tok_s_proxy(self):
        result = run_t003_receipt_kv_gate()

        self.assertEqual(result["activation"], "T003_RECEIPT_KV_GATE_ACTIVE")
        self.assertTrue(result["overall"]["pass"])
        self.assertLess(result["overall"]["t003_total_gib"], result["overall"]["baseline_total_gib"])
        self.assertGreater(result["overall"]["t003_slack_gib"], result["overall"]["baseline_slack_gib"] + 0.50)
        self.assertGreaterEqual(result["overall"]["tok_s_speedup_vs_baseline"], 1.10)
        self.assertTrue(result["validations"]["receipt_hash_present"])

    def test_t004_config_snapshot_matches_project_qwen_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )

            snapshot = resolve_qwen_config(
                model_id="Qwen/Qwen3-32B",
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
            )

        self.assertTrue(snapshot.matches_project_qwen3_32b_shape())
        self.assertEqual(snapshot.source, "local_model_dir")
        self.assertGreater(snapshot.to_shape().estimated_weight_elements, 30_000_000_000)

    def test_t004_safetensors_inventory_reads_headers_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            header = {
                "model.layers.0.mlp.down_proj.weight": {
                    "dtype": "F16",
                    "shape": [8, 16],
                    "data_offsets": [0, 256],
                },
                "model.layers.0.self_attn.q_proj.weight": {
                    "dtype": "F16",
                    "shape": [4, 16],
                    "data_offsets": [256, 384],
                },
            }
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            (model_dir / "model-00001-of-00001.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + bytes(384)
            )

            inventory = inventory_safetensors(model_dir)

        self.assertTrue(inventory.local_weights_present)
        self.assertEqual(inventory.shard_count, 1)
        self.assertEqual(inventory.tensor_count, 2)
        self.assertEqual(inventory.largest_tensor_name, "model.layers.0.mlp.down_proj.weight")

    def test_t004_bridge_classifies_missing_weights_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )

            result = run_t004_real_qwen_bridge(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
            )

        self.assertEqual(result["activation"], "T004_REAL_QWEN_BRIDGE_ACTIVE")
        self.assertTrue(result["readiness"]["bridge_pass"])
        self.assertTrue(result["readiness"]["config_matches_project_qwen3_32b_shape"])
        self.assertFalse(result["readiness"]["local_weights_present"])
        self.assertIn("missing_local_qwen3_32b_safetensors", result["blockers"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for tensor sentinel test",
    )
    def test_t005_real_weight_sentinel_samples_fake_safetensors(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            tensor_names = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            ]
            save_file(
                {
                    tensor_names[0]: torch.arange(64, dtype=torch.float32).reshape(8, 8) + 1,
                    tensor_names[1]: torch.arange(64, dtype=torch.float32).reshape(8, 8) + 2,
                },
                model_dir / "model-00001-of-00001.safetensors",
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensor_names}}),
                encoding="utf-8",
            )

            result = run_t005_real_weight_sentinel(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                tensor_names=tensor_names,
                min_required_tensors=2,
            )

        self.assertEqual(result["activation"], "T005_REAL_WEIGHT_SENTINEL_ACTIVE")
        self.assertTrue(result["readiness"]["sentinel_pass"])
        self.assertEqual(result["readiness"]["sampled_tensors"], 2)
        self.assertEqual(result["blockers"], [])
        self.assertTrue(all(item["passes"] for item in result["sentinels"]))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for block smoke test",
    )
    def test_t006_real_residency_block_smoke_compresses_fake_safetensors(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            tensor_names = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            ]
            rows = torch.linspace(-1.0, 1.0, steps=2048, dtype=torch.float32).reshape(8, 256)
            save_file(
                {
                    tensor_names[0]: rows,
                    tensor_names[1]: rows * 0.5,
                },
                model_dir / "model-00001-of-00001.safetensors",
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensor_names}}),
                encoding="utf-8",
            )

            result = run_t006_real_residency_block_smoke(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                tensor_names=tensor_names,
                block_rows=8,
                block_cols=256,
                batch=4,
                min_required_blocks=2,
            )

        self.assertEqual(result["activation"], "T006_REAL_RESIDENCY_BLOCK_SMOKE_ACTIVE")
        self.assertTrue(result["readiness"]["smoke_pass"])
        self.assertEqual(result["readiness"]["sampled_blocks"], 2)
        self.assertEqual(result["blockers"], [])
        self.assertTrue(all(item["passes"] for item in result["blocks"]))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for layer smoke test",
    )
    def test_t007_real_layer_smoke_runs_fake_safetensors(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            names = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            ]
            tensors = {
                names[0]: torch.linspace(-0.25, 0.25, steps=32 * 64, dtype=torch.float32).reshape(32, 64),
                names[1]: torch.linspace(-0.20, 0.20, steps=32 * 64, dtype=torch.float32).reshape(32, 64),
                names[2]: torch.linspace(-0.15, 0.15, steps=32 * 64, dtype=torch.float32).reshape(32, 64),
                names[3]: torch.linspace(-0.10, 0.10, steps=64 * 32, dtype=torch.float32).reshape(64, 32),
                names[4]: torch.linspace(-0.18, 0.18, steps=96 * 64, dtype=torch.float32).reshape(96, 64),
                names[5]: torch.linspace(-0.16, 0.16, steps=96 * 64, dtype=torch.float32).reshape(96, 64),
                names[6]: torch.linspace(-0.14, 0.14, steps=64 * 96, dtype=torch.float32).reshape(64, 96),
            }
            save_file(tensors, model_dir / "model-00001-of-00001.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}),
                encoding="utf-8",
            )

            result = run_t007_real_layer_residency_smoke(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                layer_index=0,
                hidden_cols=64,
                attention_width=32,
                mlp_width=96,
                sequence_tokens=4,
            )

        self.assertEqual(result["activation"], "T007_REAL_LAYER_RESIDENCY_SMOKE_ACTIVE")
        self.assertTrue(result["readiness"]["smoke_pass"])
        self.assertEqual(result["blockers"], [])
        self.assertIsNotNone(result["layer_receipt"])
        self.assertEqual(len(result["tensor_receipts"]), 7)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for compressed generation smoke test",
    )
    def test_t008_compressed_generation_smoke_runs_fake_safetensors(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            names = ["model.embed_tokens.weight", "lm_head.weight"]
            for layer_index in (0, 1):
                prefix = f"model.layers.{layer_index}"
                names.extend(
                    [
                        f"{prefix}.self_attn.q_proj.weight",
                        f"{prefix}.self_attn.k_proj.weight",
                        f"{prefix}.self_attn.v_proj.weight",
                        f"{prefix}.self_attn.o_proj.weight",
                        f"{prefix}.mlp.gate_proj.weight",
                        f"{prefix}.mlp.up_proj.weight",
                        f"{prefix}.mlp.down_proj.weight",
                    ]
                )
            tensors = {
                "model.embed_tokens.weight": torch.linspace(-0.12, 0.12, steps=48 * 64, dtype=torch.float32).reshape(48, 64),
                "lm_head.weight": torch.linspace(-0.10, 0.10, steps=48 * 64, dtype=torch.float32).reshape(48, 64),
            }
            for layer_index in (0, 1):
                prefix = f"model.layers.{layer_index}"
                scale = 1.0 / float(layer_index + 1)
                tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.linspace(-0.12, 0.12, steps=32 * 64, dtype=torch.float32).reshape(32, 64) * scale
                tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.linspace(-0.10, 0.10, steps=32 * 64, dtype=torch.float32).reshape(32, 64) * scale
                tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.linspace(-0.08, 0.08, steps=32 * 64, dtype=torch.float32).reshape(32, 64) * scale
                tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.linspace(-0.06, 0.06, steps=64 * 32, dtype=torch.float32).reshape(64, 32) * scale
                tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.linspace(-0.09, 0.09, steps=96 * 64, dtype=torch.float32).reshape(96, 64) * scale
                tensors[f"{prefix}.mlp.up_proj.weight"] = torch.linspace(-0.08, 0.08, steps=96 * 64, dtype=torch.float32).reshape(96, 64) * scale
                tensors[f"{prefix}.mlp.down_proj.weight"] = torch.linspace(-0.07, 0.07, steps=64 * 96, dtype=torch.float32).reshape(64, 96) * scale
            save_file(tensors, model_dir / "model-00001-of-00001.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}),
                encoding="utf-8",
            )

            result = run_t008_compressed_generation_smoke(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                layer_indices=(0, 1),
                hidden_cols=64,
                attention_width=32,
                mlp_width=96,
                vocab_rows=16,
                prompt_tokens=(1, 2, 3),
                max_new_tokens=3,
            )

        self.assertEqual(result["activation"], "T008_COMPRESSED_GENERATION_SMOKE_ACTIVE")
        self.assertTrue(result["readiness"]["smoke_pass"])
        self.assertEqual(result["blockers"], [])
        self.assertIsNotNone(result["generation_receipt"])
        self.assertEqual(len(result["generation_receipt"]["generated_tokens"]), 3)
        self.assertEqual(len(result["tensor_receipts"]), 16)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for runtime scaler test",
    )
    def test_t009_t013_runtime_scaler_runs_fake_safetensors(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            names = ["model.embed_tokens.weight", "lm_head.weight"]
            tensors = {
                "model.embed_tokens.weight": torch.linspace(-0.05, 0.05, steps=16 * 32, dtype=torch.float32).reshape(16, 32),
                "lm_head.weight": torch.linspace(-0.04, 0.04, steps=16 * 32, dtype=torch.float32).reshape(16, 32),
            }
            for layer_index in (0, 1):
                prefix = f"model.layers.{layer_index}"
                layer_names = [
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                ]
                names.extend(layer_names)
                scale = 1.0 / float(layer_index + 1)
                tensors[layer_names[0]] = torch.linspace(-0.12, 0.12, steps=24 * 32, dtype=torch.float32).reshape(24, 32) * scale
                tensors[layer_names[1]] = torch.linspace(-0.10, 0.10, steps=8 * 32, dtype=torch.float32).reshape(8, 32) * scale
                tensors[layer_names[2]] = torch.linspace(-0.08, 0.08, steps=8 * 32, dtype=torch.float32).reshape(8, 32) * scale
                tensors[layer_names[3]] = torch.linspace(-0.06, 0.06, steps=32 * 24, dtype=torch.float32).reshape(32, 24) * scale
                tensors[layer_names[4]] = torch.linspace(-0.09, 0.09, steps=48 * 32, dtype=torch.float32).reshape(48, 32) * scale
                tensors[layer_names[5]] = torch.linspace(-0.08, 0.08, steps=48 * 32, dtype=torch.float32).reshape(48, 32) * scale
                tensors[layer_names[6]] = torch.linspace(-0.07, 0.07, steps=32 * 48, dtype=torch.float32).reshape(32, 48) * scale
            save_file(tensors, model_dir / "model-00001-of-00001.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}),
                encoding="utf-8",
            )

            result = run_t009_t013_runtime_scaler(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                full_layer_index=0,
                scale_layer_indices=(0, 1),
                scale_row_limit=8,
                projection_batch=1,
                prompt_text="fake prompt",
                require_tokenizer=False,
                kv_context_tokens=(16, 32),
                exact_recent_tokens=4,
            )

        self.assertEqual(result["activation"], "T009_T013_QWEN_RUNTIME_SCALER_ACTIVE")
        self.assertTrue(result["readiness"]["smoke_pass"])
        self.assertEqual(result["blockers"], [])
        self.assertTrue(result["t009_full_width_layer"]["passes"])
        self.assertTrue(result["t010_multilayer_scale"]["passes"])
        self.assertTrue(result["t012_kv_cache_ledger"]["passes"])
        self.assertTrue(result["t013_full_model_equivalent_pricing"]["passes"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for T014 denominator test",
    )
    def test_t014_denominator_tightening_prices_fit_positive_stack(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            names = [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            ]
            base = torch.linspace(-0.12, 0.12, steps=12 * 512, dtype=torch.float32).reshape(12, 512)
            tensors = {
                names[0]: base,
                names[1]: base * 0.8,
                names[2]: base * 0.6,
            }
            save_file(tensors, model_dir / "model-00001-of-00001.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}),
                encoding="utf-8",
            )

            result = run_t014_denominator_tightening(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                mlp_layer_index=0,
                mlp_row_limit=8,
                projection_batch=1,
                include_stretch=False,
            )

        candidates = {candidate["name"]: candidate for candidate in result["candidate_table"]}
        self.assertEqual(result["activation"], "T014_DENOMINATOR_TIGHTENING_ACTIVE")
        self.assertTrue(result["readiness"]["t014_a_pass"])
        self.assertTrue(result["mlp_projection_gate"]["passes"])
        self.assertTrue(result["mlp_projection_gate"]["target_bits_pass"])
        self.assertTrue(result["packed_arena_receipt"]["passes"])
        self.assertFalse(result["readiness"]["full_model_text_inference_ready"])
        self.assertFalse(candidates["current_t013"]["candidate_fits_12gib"])
        self.assertTrue(candidates["t014_a_f15_mlp245_packed_overhead"]["candidate_fits_12gib"])
        self.assertTrue(candidates["t014_f21_near_term_stack"]["candidate_hits_preferred_margin"])
        self.assertLess(
            candidates["t014_f22_strong_stack"]["total_with_kv_gib"],
            candidates["t014_f21_near_term_stack"]["total_with_kv_gib"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for T015 inference test",
    )
    def test_t015_packed_qwen_inference_runs_fake_streamed_token(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            hidden = 16
            q_width = 32
            kv_width = 8
            mlp = 32
            vocab = 256
            prefix = "model.layers.0"
            names = [
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
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
            tensors = {
                "model.embed_tokens.weight": torch.linspace(-0.2, 0.2, steps=vocab * hidden, dtype=torch.float32).reshape(vocab, hidden),
                "model.norm.weight": torch.ones(hidden, dtype=torch.float32),
                "lm_head.weight": torch.linspace(-0.1, 0.1, steps=vocab * hidden, dtype=torch.float32).reshape(vocab, hidden),
                f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.self_attn.q_norm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.self_attn.k_norm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.self_attn.q_proj.weight": torch.linspace(-0.09, 0.09, steps=q_width * hidden, dtype=torch.float32).reshape(q_width, hidden),
                f"{prefix}.self_attn.k_proj.weight": torch.linspace(-0.08, 0.08, steps=kv_width * hidden, dtype=torch.float32).reshape(kv_width, hidden),
                f"{prefix}.self_attn.v_proj.weight": torch.linspace(-0.07, 0.07, steps=kv_width * hidden, dtype=torch.float32).reshape(kv_width, hidden),
                f"{prefix}.self_attn.o_proj.weight": torch.linspace(-0.06, 0.06, steps=hidden * q_width, dtype=torch.float32).reshape(hidden, q_width),
                f"{prefix}.mlp.gate_proj.weight": torch.linspace(-0.05, 0.05, steps=mlp * hidden, dtype=torch.float32).reshape(mlp, hidden),
                f"{prefix}.mlp.up_proj.weight": torch.linspace(-0.04, 0.04, steps=mlp * hidden, dtype=torch.float32).reshape(mlp, hidden),
                f"{prefix}.mlp.down_proj.weight": torch.linspace(-0.03, 0.03, steps=hidden * mlp, dtype=torch.float32).reshape(hidden, mlp),
            }
            save_file(tensors, model_dir / "model-00001-of-00001.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}),
                encoding="utf-8",
            )

            result = run_t015_packed_qwen_inference(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cpu",
                lm_head_chunk_rows=64,
            )

        self.assertEqual(result["activation"], "T015_PACKED_QWEN_STREAMED_INFERENCE_ACTIVE")
        self.assertTrue(result["readiness"]["smoke_pass"])
        self.assertFalse(result["readiness"]["full_model_text_inference_ready"])
        self.assertEqual(result["receipt"]["completed_layers"], 1)
        self.assertEqual(len(result["receipt"]["generated_token_ids"]), 1)
        self.assertEqual(result["blockers"], [])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None,
        "torch and safetensors are required for T016 persistent cache test",
    )
    def test_t016_persistent_packed_cache_reuses_fake_cache(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            hidden = 16
            q_width = 32
            kv_width = 8
            mlp = 32
            vocab = 256
            prefix = "model.layers.0"
            names = [
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
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
            tensors = {
                "model.embed_tokens.weight": torch.linspace(-0.2, 0.2, steps=vocab * hidden, dtype=torch.float32).reshape(vocab, hidden),
                "model.norm.weight": torch.ones(hidden, dtype=torch.float32),
                "lm_head.weight": torch.linspace(-0.1, 0.1, steps=vocab * hidden, dtype=torch.float32).reshape(vocab, hidden),
                f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.self_attn.q_norm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.self_attn.k_norm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.self_attn.q_proj.weight": torch.linspace(-0.09, 0.09, steps=q_width * hidden, dtype=torch.float32).reshape(q_width, hidden),
                f"{prefix}.self_attn.k_proj.weight": torch.linspace(-0.08, 0.08, steps=kv_width * hidden, dtype=torch.float32).reshape(kv_width, hidden),
                f"{prefix}.self_attn.v_proj.weight": torch.linspace(-0.07, 0.07, steps=kv_width * hidden, dtype=torch.float32).reshape(kv_width, hidden),
                f"{prefix}.self_attn.o_proj.weight": torch.linspace(-0.06, 0.06, steps=hidden * q_width, dtype=torch.float32).reshape(hidden, q_width),
                f"{prefix}.mlp.gate_proj.weight": torch.linspace(-0.05, 0.05, steps=mlp * hidden, dtype=torch.float32).reshape(mlp, hidden),
                f"{prefix}.mlp.up_proj.weight": torch.linspace(-0.04, 0.04, steps=mlp * hidden, dtype=torch.float32).reshape(mlp, hidden),
                f"{prefix}.mlp.down_proj.weight": torch.linspace(-0.03, 0.03, steps=hidden * mlp, dtype=torch.float32).reshape(hidden, mlp),
            }
            save_file(tensors, model_dir / "model-00001-of-00001.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}),
                encoding="utf-8",
            )
            cache_dir = Path(tmp) / "cache"

            first = run_t016_persistent_packed_cache(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cpu",
                lm_head_chunk_rows=64,
                cache_dir=cache_dir,
                refresh_cache=True,
            )
            second = run_t016_persistent_packed_cache(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cpu",
                lm_head_chunk_rows=64,
                cache_dir=cache_dir,
            )

        self.assertEqual(first["activation"], "T016_PERSISTENT_PACKED_CACHE_ACTIVE")
        self.assertTrue(first["readiness"]["smoke_pass"])
        self.assertTrue(second["readiness"]["smoke_pass"])
        self.assertGreater(first["receipt"]["cache_stats"]["cache_build_count"], 0)
        self.assertEqual(second["receipt"]["cache_stats"]["cache_build_count"], 0)
        self.assertGreater(second["receipt"]["cache_stats"]["cache_hit_count"], 0)
        self.assertEqual(first["receipt"]["generated_token_ids"], second["receipt"]["generated_token_ids"])
        self.assertEqual(first["blockers"], [])
        self.assertEqual(second["blockers"], [])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None and importlib.util.find_spec("triton") is not None,
        "torch and triton are required for T017 packed linear kernel test",
    )
    def test_t017_packed_linear_kernel_gate_runs(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for T017 packed linear kernel test")

        result = run_t017_packed_linear_kernel_gate(
            rows=64,
            cols=64,
            tokens=2,
            repair_fraction=0.03,
            iterations=3,
            seed=17017,
        )

        self.assertEqual(result["activation"], "T017_PACKED_LINEAR_KERNEL_ACTIVE")
        self.assertTrue(result["readiness"]["kernel_gate_pass"], result["blockers"])
        self.assertEqual(result["blockers"], [])

    def test_t020_device_credit_flow_gate_prices_overlap_and_memory(self):
        result = run_t020_device_credit_flow_gate()

        self.assertEqual(result["activation"], "T020_DEVICE_CREDIT_FLOW_ACTIVE")
        self.assertTrue(result["readiness"]["device_credit_flow_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["memory_credit_fits_12gib"])
        self.assertTrue(result["readiness"]["gain_floor_pass"])
        self.assertGreaterEqual(result["candidate"]["speedup_vs_baseline"], 1.10)
        self.assertLess(
            result["baseline"]["denominator_total_gib"] + result["candidate"]["live_credit_gib"],
            result["baseline"]["device_budget_gib"],
        )
        self.assertEqual(result["blockers"], [])

    def test_t020_packed_work_coalescing_gate_maps_residual_destinations(self):
        result = run_t020_packed_work_coalescing_gate()

        self.assertEqual(result["activation"], "T020_PACKED_WORK_COALESCING_ACTIVE")
        self.assertTrue(result["readiness"]["packed_work_coalescing_gate_ready"], result["blockers"])
        self.assertEqual(result["readiness"]["legal_outputs_per_block"], 2)
        self.assertEqual(result["legal_boundaries"], ["self_attn.o_proj", "mlp.down_proj"])
        self.assertGreater(result["traffic"]["deleted_dense_mib"], 0.0)
        self.assertEqual(result["blockers"], [])

    def test_t020_execution_order_slab_gate_reduces_acquisition_units(self):
        result = run_t020_execution_order_slab_gate()

        self.assertEqual(result["activation"], "T020_EXECUTION_ORDER_SLAB_ARENA_ACTIVE")
        self.assertTrue(result["readiness"]["execution_order_slab_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["same_payload_object_count"])
        self.assertEqual(result["baseline"]["objects"], 82)
        self.assertEqual(result["candidate"]["objects"], 1)
        self.assertEqual(result["candidate"]["offset_entries"], 82)
        self.assertGreaterEqual(result["candidate"]["acquisition_speedup"], 2.0)
        self.assertEqual(result["blockers"], [])

    def test_t020_immutable_arena_receipt_gate_binds_hot_path_identity(self):
        result = run_t020_immutable_arena_receipt_gate()

        self.assertEqual(result["activation"], "T020_IMMUTABLE_ARENA_RECEIPT_ACTIVE")
        self.assertTrue(result["readiness"]["immutable_arena_receipt_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["required_bindings_complete"])
        self.assertTrue(result["readiness"]["descriptor_root_resolves_without_payload_load"])
        self.assertEqual(result["candidate"]["descriptor_count"], 82)
        self.assertFalse(result["candidate"]["requires_original_payload_objects"])
        self.assertGreaterEqual(result["candidate"]["speedup_vs_dynamic_discovery"], 1.10)
        self.assertEqual(result["blockers"], [])

    def test_t020_conversion_time_arena_compiler_gate_proves_descriptor_equality(self):
        result = run_t020_conversion_time_arena_compiler_gate()

        self.assertEqual(result["activation"], "T020_CONVERSION_TIME_ARENA_COMPILER_ACTIVE")
        self.assertTrue(result["readiness"]["conversion_time_arena_compiler_gate_ready"], result["blockers"])
        self.assertTrue(result["descriptor_equality"]["compiled_equals_runtime_interpreter"])
        self.assertTrue(result["readiness"]["compile_time_dependencies_safe"])
        self.assertGreaterEqual(result["candidate"]["saved_fraction"], 0.10)
        self.assertFalse(result["candidate"]["runtime_interpreter_required"])
        self.assertEqual(result["blockers"], [])

    def test_t020_affine_packed_view_gate_matches_materialized_values(self):
        result = run_t020_affine_packed_view_gate()

        self.assertEqual(result["activation"], "T020_AFFINE_PACKED_VIEW_ACTIVE")
        self.assertTrue(result["readiness"]["affine_packed_view_gate_ready"], result["blockers"])
        self.assertTrue(result["equality"]["packed_value_equals_materialized"])
        self.assertEqual(result["equality"]["mismatch_count"], 0)
        self.assertEqual(result["equality"]["max_abs_error"], 0.0)
        self.assertGreaterEqual(result["candidate"]["saved_fraction"], 0.10)
        self.assertEqual(result["blockers"], [])

    def test_t020_recompute_on_reject_gate_defers_most_branch_positions(self):
        result = run_t020_recompute_on_reject_gate()

        self.assertEqual(result["activation"], "T020_RECOMPUTE_ON_REJECT_ACTIVE")
        self.assertTrue(result["readiness"]["recompute_on_reject_gate_ready"], result["blockers"])
        self.assertLessEqual(result["candidate"]["stored_fraction"], 0.80)
        self.assertGreater(result["candidate"]["deferred_positions"], result["candidate"]["stored_positions"])
        self.assertGreaterEqual(result["candidate"]["speedup_vs_eager"], 1.10)
        self.assertEqual(result["blockers"], [])

    def test_t020_expected_accepted_span_gate_selects_legal_span(self):
        result = run_t020_expected_accepted_span_gate()

        self.assertEqual(result["activation"], "T020_EXPECTED_ACCEPTED_SPAN_ACTIVE")
        self.assertTrue(result["readiness"]["expected_accepted_span_gate_ready"], result["blockers"])
        self.assertGreater(result["candidate"]["span_length"], 1)
        self.assertGreaterEqual(result["candidate"]["speedup_vs_one_token"], 1.10)
        collapse = next(profile for profile in result["profiles"] if profile["profile"] == "collapse_after_first")
        self.assertEqual(collapse["selected"]["span_length"], 1)
        self.assertEqual(result["blockers"], [])

    def test_t020_prefix_transaction_ledger_gate_matches_serial_prefixes(self):
        result = run_t020_prefix_transaction_ledger_gate()

        self.assertEqual(result["activation"], "T020_PREFIX_TRANSACTION_LEDGER_ACTIVE")
        self.assertTrue(result["readiness"]["prefix_transaction_ledger_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["all_prefixes_match_serial"])
        self.assertTrue(result["readiness"]["no_rejected_deltas_reachable"])
        self.assertTrue(result["readiness"]["no_compaction_required_before_next_token"])
        self.assertGreaterEqual(result["candidate"]["speedup_vs_eager_mutation"], 1.10)
        self.assertEqual(result["blockers"], [])

    def test_t020_acceptance_weighted_kv_materialization_gate_prices_delay(self):
        result = run_t020_acceptance_weighted_kv_materialization_gate()

        self.assertEqual(result["activation"], "T020_ACCEPTANCE_WEIGHTED_KV_MATERIALIZATION_ACTIVE")
        self.assertTrue(result["readiness"]["acceptance_weighted_kv_materialization_gate_ready"], result["blockers"])
        self.assertTrue(result["exactness"]["delayed_equals_serial"])
        self.assertFalse(result["exactness"]["rejected_receipts_reachable"])
        self.assertGreaterEqual(result["candidate"]["speedup_vs_eager_materialization"], 1.10)
        self.assertTrue(result["readiness"]["full_acceptance_not_overclaimed"])
        self.assertEqual(result["blockers"], [])

    def test_t020_basis_shared_kv_transform_gate_preserves_dot_products(self):
        result = run_t020_basis_shared_kv_transform_gate()

        self.assertEqual(result["activation"], "T020_BASIS_SHARED_KV_TRANSFORM_ACTIVE")
        self.assertTrue(result["readiness"]["basis_shared_kv_transform_gate_ready"], result["blockers"])
        self.assertTrue(result["proof"]["dot_product_preserved"])
        self.assertTrue(result["readiness"]["no_dense_inverse"])
        self.assertTrue(result["readiness"]["no_dual_canonical_storage"])
        self.assertGreaterEqual(result["candidate"]["speedup_vs_representation_crossing"], 1.10)
        self.assertEqual(result["blockers"], [])

    def test_t020_ring_exact_tail_receipts_gate_declares_adversarial_debt(self):
        result = run_t020_ring_exact_tail_receipts_gate()

        self.assertEqual(result["activation"], "T020_RING_EXACT_TAIL_RECEIPTS_ACTIVE")
        self.assertTrue(result["readiness"]["ring_exact_tail_receipts_gate_ready"], result["blockers"])
        self.assertGreaterEqual(result["candidate"]["exact_ring_tokens"], 2048)
        self.assertGreaterEqual(result["candidate"]["speedup_vs_full_active_context"], 1.10)
        far_anchor = next(row for row in result["patterns"] if row["pattern"] == "far_anchor_retrieval")
        self.assertTrue(far_anchor["declared_debt_required"])
        self.assertEqual(result["blockers"], [])

    def test_t020_cumulative_execution_shape_gate_declares_stack_speedup(self):
        result = run_t020_cumulative_execution_shape_gate()

        self.assertEqual(result["activation"], "T020_CUMULATIVE_EXECUTION_SHAPE_ACTIVE")
        self.assertTrue(result["readiness"]["cumulative_execution_shape_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["declared_stack_is_cumulative"])
        self.assertEqual(len(result["numeric_items"]), 11)
        self.assertEqual(len(result["structural_items"]), 1)
        product = 1.0
        for item in result["numeric_items"]:
            self.assertGreaterEqual(item["speedup"], 1.10, item["label"])
            product *= item["speedup"]
        self.assertAlmostEqual(result["composition"]["cumulative_speedup"], product)
        self.assertGreater(result["composition"]["cumulative_speedup"], 1000.0)
        self.assertGreater(result["structural_items"][0]["deleted_dense_mib"], 0.0)
        self.assertEqual(result["blockers"], [])

    def test_t021_runtime_obligation_measure_gate_licenses_t020_components(self):
        result = run_t021_runtime_obligation_measure_gate()

        self.assertEqual(result["activation"], "T021_RUNTIME_OBLIGATION_MEASURE_ACTIVE")
        self.assertTrue(result["readiness"]["runtime_obligation_measure_gate_ready"], result["blockers"])
        self.assertEqual(result["candidate"]["licensed_component_count"], 12)
        self.assertEqual(result["candidate"]["unresolved_overlap_classes"], 0)
        self.assertTrue(result["readiness"]["event_titles_unique"])
        self.assertFalse(result["candidate"]["production_speed_claim_allowed"])
        self.assertEqual(result["blockers"], [])

    def test_t021_transaction_receipt_gate_matches_serial_visibility(self):
        result = run_t021_transaction_receipt_gate()

        self.assertEqual(result["activation"], "T021_TRANSACTION_RECEIPT_GATE_ACTIVE")
        self.assertTrue(result["readiness"]["transaction_receipt_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["all_prefixes_match_serial"])
        self.assertTrue(result["readiness"]["no_rejected_suffix_reachable"])
        self.assertTrue(result["readiness"]["no_physical_compaction_required"])
        self.assertGreaterEqual(result["candidate"]["speedup_vs_eager_transaction"], 1.10)
        self.assertEqual(result["blockers"], [])

    def test_t021_page_activation_certificate_gate_keeps_far_anchor_active(self):
        result = run_t021_page_activation_certificate_gate()

        self.assertEqual(result["activation"], "T021_PAGE_ACTIVATION_CERTIFICATE_ACTIVE")
        self.assertTrue(result["readiness"]["page_activation_certificate_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["no_far_anchor_skip"])
        self.assertTrue(result["readiness"]["all_certificate_bounds_pass"])
        self.assertLess(result["candidate"]["average_tail_active_fraction"], 0.80)
        self.assertGreaterEqual(result["candidate"]["denominator_speedup_vs_dense_tail"], 1.10)
        self.assertEqual(result["blockers"], [])

    def test_t021_cumulative_decoder_runtime_path_gate_composes_ordered_gates(self):
        result = run_t021_cumulative_decoder_runtime_path_gate()

        self.assertEqual(result["activation"], "T021_CUMULATIVE_DECODER_RUNTIME_PATH_ACTIVE")
        self.assertTrue(result["readiness"]["cumulative_decoder_runtime_path_gate_ready"], result["blockers"])
        self.assertTrue(result["readiness"]["t020_proxy_replaced_by_ordered_gates"])
        self.assertEqual(result["candidate"]["runtime_obligation_components_licensed"], 12)
        self.assertTrue(result["candidate"]["transaction_receipt_gate_ready"])
        self.assertTrue(result["candidate"]["page_certificate_gate_ready"])
        self.assertGreater(result["candidate"]["non_overlapping_denominator_projection"], 1.10)
        self.assertFalse(result["candidate"]["production_tok_s_claim_allowed"])
        self.assertEqual(result["blockers"], [])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("safetensors") is not None
        and importlib.util.find_spec("triton") is not None,
        "torch, safetensors, and triton are required for T017 Qwen kernel test",
    )
    def test_t017_kernel_qwen_inference_runs_fake_token(self):
        import torch
        from safetensors.torch import save_file

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for T017 Qwen kernel test")

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "architectures": ["Qwen3ForCausalLM"],
                        "num_hidden_layers": 64,
                        "hidden_size": 5120,
                        "intermediate_size": 25600,
                        "num_attention_heads": 64,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "vocab_size": 151936,
                        "max_position_embeddings": 40960,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            hidden = 16
            q_width = 32
            kv_width = 8
            mlp = 32
            vocab = 256
            prefix = "model.layers.0"
            names = [
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
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
            tensors = {
                "model.embed_tokens.weight": torch.linspace(-0.2, 0.2, steps=vocab * hidden, dtype=torch.float32).reshape(vocab, hidden),
                "model.norm.weight": torch.ones(hidden, dtype=torch.float32),
                "lm_head.weight": torch.linspace(-0.1, 0.1, steps=vocab * hidden, dtype=torch.float32).reshape(vocab, hidden),
                f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
                f"{prefix}.self_attn.q_norm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.self_attn.k_norm.weight": torch.ones(8, dtype=torch.float32),
                f"{prefix}.self_attn.q_proj.weight": torch.linspace(-0.09, 0.09, steps=q_width * hidden, dtype=torch.float32).reshape(q_width, hidden),
                f"{prefix}.self_attn.k_proj.weight": torch.linspace(-0.08, 0.08, steps=kv_width * hidden, dtype=torch.float32).reshape(kv_width, hidden),
                f"{prefix}.self_attn.v_proj.weight": torch.linspace(-0.07, 0.07, steps=kv_width * hidden, dtype=torch.float32).reshape(kv_width, hidden),
                f"{prefix}.self_attn.o_proj.weight": torch.linspace(-0.06, 0.06, steps=hidden * q_width, dtype=torch.float32).reshape(hidden, q_width),
                f"{prefix}.mlp.gate_proj.weight": torch.linspace(-0.05, 0.05, steps=mlp * hidden, dtype=torch.float32).reshape(mlp, hidden),
                f"{prefix}.mlp.up_proj.weight": torch.linspace(-0.04, 0.04, steps=mlp * hidden, dtype=torch.float32).reshape(mlp, hidden),
                f"{prefix}.mlp.down_proj.weight": torch.linspace(-0.03, 0.03, steps=hidden * mlp, dtype=torch.float32).reshape(hidden, mlp),
            }
            save_file(tensors, model_dir / "model-00001-of-00001.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}),
                encoding="utf-8",
            )

            result = run_t017_kernel_qwen_inference(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            coalesced = run_t017_kernel_qwen_inference(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
                packed_work_coalescing=True,
            )
            resident = run_t018_resident_payload_qwen_inference(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
                preload_payloads=True,
            )
            binary_first = run_t019_binary_payload_qwen_inference(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            binary_hot = run_t019_binary_payload_qwen_inference(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t021_runtime = run_t021_qwen_runtime(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t022_runtime = run_t022_qwen_runtime(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t023_runtime = run_t023_qwen_runtime(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t024_runtime = run_t024_qwen_runtime(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t025_runtime = run_t025_qwen_runtime(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t026_runtime = run_t026_qwen_runtime(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t027_runtime = run_t027_qwen_runtime(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t028_runtime = run_t028_native_generation_speed(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t029_runtime = run_t029_speed_gpu_efficiency(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t030_runtime = run_t030_speed_gpu_efficiency(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )
            t031_runtime = run_t031_macro_architecture_stack(
                local_model_dir=model_dir,
                allow_hf_config_fetch=False,
                prompt_text="A",
                max_new_tokens=1,
                layer_limit=1,
                device="cuda",
                lm_head_chunk_rows=64,
                cache_dir=Path(tmp) / "cache",
            )

        self.assertEqual(result["activation"], "T017_PACKED_LINEAR_KERNEL_ACTIVE")
        self.assertTrue(result["readiness"]["smoke_pass"], result["blockers"])
        self.assertEqual(result["receipt"]["completed_layers"], 1)
        self.assertEqual(len(result["receipt"]["generated_token_ids"]), 1)
        self.assertEqual(result["blockers"], [])
        self.assertEqual(coalesced["activation"], "T017_PACKED_LINEAR_KERNEL_ACTIVE")
        self.assertTrue(coalesced["readiness"]["smoke_pass"], coalesced["blockers"])
        self.assertTrue(coalesced["readiness"]["packed_work_coalescing_gate_ready"])
        self.assertEqual(coalesced["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertEqual(coalesced["receipt"]["packed_work_coalescing"]["legal_boundaries"], ["self_attn.o_proj", "mlp.down_proj"])
        self.assertEqual(coalesced["blockers"], [])
        self.assertEqual(resident["activation"], "T018_RESIDENT_PAYLOAD_RUNTIME_ACTIVE")
        self.assertTrue(resident["readiness"]["smoke_pass"], resident["blockers"])
        self.assertTrue(resident["receipt"]["resident_payload_cache"])
        self.assertTrue(resident["receipt"]["preload_payloads"])
        self.assertGreater(resident["receipt"]["cache_stats"]["resident_stores"], 0)
        self.assertGreater(resident["receipt"]["cache_stats"]["resident_hits"], 0)
        self.assertEqual(resident["blockers"], [])
        self.assertEqual(binary_first["activation"], "T019_BINARY_PAYLOAD_RUNTIME_ACTIVE")
        self.assertTrue(binary_first["readiness"]["smoke_pass"], binary_first["blockers"])
        self.assertGreater(binary_first["receipt"]["cache_stats"]["binary_payloads_written"], 0)
        self.assertEqual(binary_first["blockers"], [])
        self.assertEqual(binary_hot["activation"], "T019_BINARY_PAYLOAD_RUNTIME_ACTIVE")
        self.assertTrue(binary_hot["readiness"]["smoke_pass"], binary_hot["blockers"])
        self.assertGreater(binary_hot["receipt"]["cache_stats"]["binary_payloads_loaded"], 0)
        self.assertEqual(binary_hot["blockers"], [])
        self.assertEqual(t021_runtime["activation"], "T021_QWEN_RUNTIME_ACTIVE")
        self.assertTrue(t021_runtime["readiness"]["smoke_pass"], t021_runtime["blockers"])
        self.assertTrue(t021_runtime["readiness"]["t021_cumulative_gate_ready"])
        self.assertTrue(t021_runtime["readiness"]["packed_work_coalescing_gate_ready"])
        self.assertTrue(t021_runtime["receipt"]["resident_payload_cache"])
        self.assertEqual(t021_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        if t021_runtime["readiness"]["t021_qwen_runtime_ready"]:
            self.assertTrue(t021_runtime["readiness"]["device_credit_flow_gate_ready"])
            self.assertTrue(t021_runtime["readiness"]["t021_runtime_payload_reuse_ready"])
            self.assertEqual(t021_runtime["blockers"], [])
        else:
            self.assertIn("t021_qwen_runtime_not_ready", t021_runtime["blockers"])
        self.assertEqual(t022_runtime["activation"], "T022_QWEN_RUNTIME_ACTIVE")
        self.assertTrue(t022_runtime["readiness"]["smoke_pass"], t022_runtime["blockers"])
        self.assertTrue(t022_runtime["readiness"]["t022_qwen_runtime_ready"], t022_runtime)
        self.assertTrue(t022_runtime["receipt"]["device_payload_cache"])
        self.assertTrue(t022_runtime["receipt"]["preload_device_payloads"])
        self.assertTrue(t022_runtime["receipt"]["dense_tensor_cache"])
        self.assertTrue(t022_runtime["readiness"]["t022_device_credit_flow_receipt_present"])
        self.assertGreater(t022_runtime["receipt"]["cache_stats"]["device_resident_stores"], 0)
        self.assertGreater(t022_runtime["receipt"]["cache_stats"]["device_resident_hits"], 0)
        self.assertEqual(t022_runtime["receipt"]["cache_stats"]["device_resident_failures"], 0)
        self.assertEqual(t022_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertEqual(t022_runtime["blockers"], [])
        self.assertEqual(t023_runtime["activation"], "T023_QWEN_RUNTIME_ACTIVE")
        self.assertTrue(t023_runtime["readiness"]["smoke_pass"], t023_runtime["blockers"])
        self.assertTrue(t023_runtime["readiness"]["t023_qwen_runtime_ready"], t023_runtime)
        self.assertTrue(t023_runtime["readiness"]["t023_no_cuda_payload_residency"])
        self.assertTrue(t023_runtime["receipt"]["dense_tensor_cache"])
        self.assertFalse(t023_runtime["receipt"]["device_payload_cache"])
        self.assertFalse(t023_runtime["receipt"]["preload_device_payloads"])
        self.assertEqual(t023_runtime["receipt"]["cache_stats"]["device_resident_bytes"], 0)
        self.assertEqual(t023_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertEqual(t023_runtime["blockers"], [])
        self.assertEqual(t024_runtime["activation"], "T024_QWEN_RUNTIME_ACTIVE")
        self.assertTrue(t024_runtime["readiness"]["smoke_pass"], t024_runtime["blockers"])
        self.assertTrue(t024_runtime["readiness"]["t024_qwen_runtime_ready"], t024_runtime)
        self.assertTrue(t024_runtime["readiness"]["t024_no_cuda_payload_residency"])
        self.assertTrue(t024_runtime["readiness"]["t024_inplace_mlp_intermediate_ready"])
        self.assertTrue(t024_runtime["receipt"]["dense_tensor_cache"])
        self.assertFalse(t024_runtime["receipt"]["device_payload_cache"])
        self.assertFalse(t024_runtime["receipt"]["preload_device_payloads"])
        self.assertEqual(t024_runtime["receipt"]["cache_stats"]["device_resident_bytes"], 0)
        self.assertEqual(t024_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertIn("peak_process_working_set_gib", t024_runtime["receipt"])
        self.assertEqual(t024_runtime["blockers"], [])
        self.assertEqual(t025_runtime["activation"], "T025_QWEN_RUNTIME_ACTIVE")
        self.assertTrue(t025_runtime["readiness"]["smoke_pass"], t025_runtime["blockers"])
        self.assertTrue(t025_runtime["readiness"]["t025_qwen_runtime_ready"], t025_runtime)
        self.assertTrue(t025_runtime["readiness"]["t025_fail_closed_no_miss_ready"])
        self.assertTrue(t025_runtime["readiness"]["t025_hot_payload_load_free"])
        self.assertTrue(t025_runtime["readiness"]["t025_real_gpu_payload_residency"])
        self.assertTrue(t025_runtime["receipt"]["device_payload_cache"])
        self.assertTrue(t025_runtime["receipt"]["preload_device_payloads"])
        self.assertTrue(t025_runtime["receipt"]["trit_packed_codes"])
        self.assertTrue(t025_runtime["receipt"]["fail_closed_device_payloads"])
        self.assertGreater(t025_runtime["receipt"]["cache_stats"]["device_resident_bytes"], 0)
        self.assertGreater(t025_runtime["receipt"]["cache_stats"]["device_arena_hits"], 0)
        self.assertEqual(t025_runtime["receipt"]["cache_stats"]["device_arena_misses"], 0)
        self.assertGreater(len(t025_runtime["receipt"]["device_payload_descriptor_table"]), 0)
        self.assertEqual(t025_runtime["receipt"]["hot_payload_load_events"], 0)
        self.assertEqual(t025_runtime["receipt"]["hot_payload_load_seconds"], 0.0)
        self.assertEqual(t025_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertEqual(t025_runtime["blockers"], [])
        self.assertEqual(t026_runtime["activation"], "T026_QWEN_RUNTIME_ACTIVE")
        self.assertTrue(t026_runtime["readiness"]["smoke_pass"], t026_runtime["blockers"])
        self.assertTrue(t026_runtime["readiness"]["t026_qwen_runtime_ready"], t026_runtime)
        self.assertTrue(t026_runtime["readiness"]["t026_deferred_sync_ready"])
        self.assertFalse(t026_runtime["readiness"]["t026_gpu_argmax_head_ready"])
        self.assertTrue(t026_runtime["readiness"]["t026_gpu_argmax_head_rejected"])
        self.assertTrue(t026_runtime["readiness"]["t026_fail_closed_no_miss_ready"])
        self.assertTrue(t026_runtime["readiness"]["t026_hot_payload_load_free"])
        self.assertTrue(t026_runtime["readiness"]["t026_real_gpu_payload_residency"])
        self.assertTrue(t026_runtime["receipt"]["defer_kernel_synchronization"])
        self.assertFalse(t026_runtime["receipt"]["gpu_argmax_head"])
        self.assertEqual(t026_runtime["receipt"]["kernel_timing_mode"], "deferred_submit")
        self.assertEqual(t026_runtime["receipt"]["cache_stats"]["device_arena_misses"], 0)
        self.assertEqual(t026_runtime["receipt"]["hot_payload_load_events"], 0)
        self.assertEqual(t026_runtime["receipt"]["hot_payload_load_seconds"], 0.0)
        self.assertEqual(t026_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertEqual(t026_runtime["blockers"], [])
        self.assertEqual(t027_runtime["activation"], "T027_GENERATION_SPEED_STACK_ACTIVE")
        self.assertTrue(t027_runtime["readiness"]["smoke_pass"], t027_runtime["blockers"])
        self.assertTrue(t027_runtime["readiness"]["t027_qwen_runtime_ready"], t027_runtime)
        self.assertTrue(t027_runtime["readiness"]["t027_cumulative_over_t026"])
        self.assertTrue(t027_runtime["readiness"]["t027_deferred_sync_ready"])
        self.assertTrue(t027_runtime["readiness"]["t027_gpu_greedy_tournament_ready"])
        self.assertTrue(t027_runtime["readiness"]["t027_fused_mlp_gate_up_ready"])
        self.assertTrue(t027_runtime["readiness"]["t027_hot_payload_load_free"])
        self.assertTrue(t027_runtime["readiness"]["t027_real_gpu_payload_residency"])
        self.assertTrue(t027_runtime["receipt"]["defer_kernel_synchronization"])
        self.assertTrue(t027_runtime["receipt"]["gpu_argmax_head"])
        self.assertTrue(t027_runtime["receipt"]["fused_mlp_gate_up_kernel"])
        self.assertEqual(t027_runtime["receipt"]["kernel_timing_mode"], "deferred_submit")
        self.assertEqual(t027_runtime["receipt"]["cache_stats"]["device_arena_misses"], 0)
        self.assertEqual(t027_runtime["receipt"]["hot_payload_load_events"], 0)
        self.assertEqual(t027_runtime["receipt"]["hot_payload_load_seconds"], 0.0)
        self.assertEqual(t027_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertFalse(t027_runtime["t027_projection"]["validated_continuous_generation"]["robustly_crosses_40_tok_s"])
        self.assertGreater(t027_runtime["t027_speed_comparison"]["measured_tokens_per_second"], 0.0)
        self.assertEqual(t027_runtime["blockers"], [])
        self.assertEqual(t028_runtime["activation"], "T028_NATIVE_GENERATION_SPEED_ACTIVE")
        self.assertTrue(t028_runtime["readiness"]["smoke_pass"], t028_runtime["blockers"])
        self.assertTrue(t028_runtime["readiness"]["t028_denominator_ledger_ready"])
        self.assertTrue(t028_runtime["readiness"]["t028_fixed_address_graph_contract_ready"])
        self.assertTrue(t028_runtime["readiness"]["t028_device_argmax_embedding_contract_ready"])
        self.assertTrue(t028_runtime["readiness"]["t028_stable_virtual_kv_contract_ready"])
        self.assertTrue(t028_runtime["readiness"]["t028_macro_phase_graph_contract_ready"])
        self.assertTrue(t028_runtime["readiness"]["t028_whole_token_program_substrate_ready"])
        self.assertTrue(t028_runtime["readiness"]["t028_certificate_density_contract_ready"])
        if t028_runtime["readiness"]["t028_native_compact_affine_contract_ready"]:
            self.assertTrue(t028_runtime["readiness"]["t028_native_generation_speed_ready"], t028_runtime)
        else:
            self.assertFalse(t028_runtime["readiness"]["t028_native_generation_speed_ready"])
            self.assertIn("t028_native_generation_speed_gate_not_ready", t028_runtime["blockers"])
        self.assertTrue(t028_runtime["receipt"]["prompt_embeddings_preloaded"])
        self.assertTrue(t028_runtime["readiness"]["t027_prompt_embedding_preload_ready"])
        self.assertTrue(t028_runtime["receipt"]["native_attention_sdp"])
        self.assertTrue(t028_runtime["readiness"]["t027_native_attention_sdp_ready"])
        self.assertFalse(t028_runtime["readiness"]["t028_production_tok_s_claim_allowed"])
        self.assertEqual(t028_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertEqual(t028_runtime["t028_gates"]["denominator_ledger"]["ledger"]["hot_payload_load_events"], 0)
        self.assertTrue(t028_runtime["t028_gates"]["denominator_ledger"]["ledger"]["hot_payload_load_free"])
        self.assertTrue(t028_runtime["t028_gates"]["fixed_address_graph_replay"]["cuda_graph_captured"])
        self.assertTrue(t028_runtime["t028_gates"]["device_argmax_embedding_closure"]["next_embedding_device_closure_proven"])
        self.assertTrue(t028_runtime["t028_gates"]["stable_virtual_kv_window"]["page_boundary_crossed_without_host_update"])
        self.assertTrue(t028_runtime["t028_gates"]["macro_phase_graph"]["four_layer_output_equality_proven"])
        self.assertTrue(t028_runtime["t028_gates"]["whole_token_program_substrate"]["multi_token_device_chunk_ready"])
        self.assertTrue(t028_runtime["t028_gates"]["native_compact_affine_tile"]["native_tile_microkernel_proven"])
        self.assertTrue(t028_runtime["t028_gates"]["native_compact_affine_tile"]["streaming_mlp_contraction_proven"])
        self.assertTrue(t028_runtime["t028_gates"]["native_compact_affine_tile"]["attention_streaming_boundary_proven"])
        self.assertTrue(t028_runtime["t028_gates"]["certificate_density"]["density_proven"])
        self.assertEqual(t028_runtime["t028_stack"]["items_not_yet_proven"], [])
        self.assertFalse(t028_runtime["t028_projection"]["production_tok_s_claim_allowed"])
        self.assertGreater(t028_runtime["t028_speed_comparison"]["measured_tokens_per_second"], 0.0)
        self.assertEqual(t028_runtime["blockers"], [])
        self.assertEqual(t029_runtime["activation"], "T029_SPEED_GPU_EFFICIENCY_ACTIVE")
        self.assertTrue(t029_runtime["readiness"]["smoke_pass"], t029_runtime["blockers"])
        self.assertTrue(t029_runtime["readiness"]["t029_speed_gpu_efficiency_ready"], t029_runtime)
        self.assertTrue(t029_runtime["readiness"]["t029_exact_20_trit_payloads_active"])
        self.assertEqual(t029_runtime["receipt"]["trit_code_group_size"], 5)
        self.assertEqual(t029_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertGreaterEqual(t029_runtime["t029_memory_audit"]["code_bytes_saved_vs_group4"], 0)
        self.assertGreater(t029_runtime["t029_speed_efficiency_comparison"]["measured_tokens_per_second"], 0.0)
        self.assertEqual(t029_runtime["blockers"], [])
        self.assertEqual(t030_runtime["activation"], "T030_SPEED_GPU_EFFICIENCY_ACTIVE")
        self.assertTrue(t030_runtime["readiness"]["smoke_pass"], t030_runtime["blockers"])
        self.assertTrue(t030_runtime["readiness"]["t030_gpu_bound_no_offload_ready"])
        self.assertTrue(t030_runtime["readiness"]["t030_all_top10_items_accounted"])
        self.assertEqual(t030_runtime["receipt"]["trit_code_group_size"], 5)
        self.assertEqual(t030_runtime["receipt"]["generated_token_ids"], result["receipt"]["generated_token_ids"])
        self.assertTrue(t030_runtime["t030_gate_audit"]["activation_codeword_direct_5trit_microdot"]["ready"])
        self.assertTrue(t030_runtime["t030_gate_audit"]["same_input_projection_bundle"]["ready"])
        self.assertTrue(t030_runtime["t030_gate_audit"]["resident_descriptor_tape"]["ready"])
        if t030_runtime["t030_gate_audit"]["mlp_supernode"]["ready"]:
            self.assertTrue(t030_runtime["readiness"]["t030_speed_gpu_efficiency_ready"], t030_runtime)
        else:
            self.assertFalse(t030_runtime["readiness"]["t030_speed_gpu_efficiency_ready"])
            self.assertGreater(len(t030_runtime["blockers"]), 0)
        self.assertTrue(t030_runtime["t030_gate_audit"]["attention_qkv_kv_supernode"]["ready"])
        self.assertTrue(t030_runtime["t030_gate_audit"]["consumer_transposed_layout"]["ready"])
        self.assertTrue(t030_runtime["t030_gate_audit"]["static_instruction_families"]["ready"])
        self.assertIn("t030_gm_v02_exact_tile_quotient_metadata_gate", t030_runtime["t030_stack"]["items_active"])
        self.assertFalse(t030_runtime["receipt"]["fused_qkv_compact_projection_kernel"])
        self.assertFalse(t030_runtime["t030_gate_audit"]["same_input_projection_bundle"]["qkv_triple_kernel_active"])
        self.assertIn("qkv_single_compact_projection_kernel", t030_runtime["t030_stack"]["items_not_integrated_into_hot_loop"])
        self.assertGreater(t030_runtime["t030_speed_efficiency_comparison"]["measured_tokens_per_second"], 0.0)
        self.assertEqual(
            t030_runtime["t030_speed_efficiency_comparison"]["comparison_scope"],
            "non_comparable_layer_limited_smoke",
        )
        self.assertIsNone(t030_runtime["t030_speed_efficiency_comparison"]["measured_speedup_vs_r452_t028"])
        self.assertFalse(t030_runtime["readiness"]["t030_production_tok_s_claim_allowed"])
        if t030_runtime["readiness"]["t030_speed_gpu_efficiency_ready"]:
            self.assertEqual(t030_runtime["blockers"], [])
        self.assertEqual(t031_runtime["activation"], "T031_MACRO_ARCHITECTURE_STACK_ACTIVE")
        self.assertTrue(t031_runtime["readiness"]["t031_real_runtime_executed"])
        self.assertTrue(t031_runtime["readiness"]["t031_real_runtime_smoke_pass"])
        self.assertFalse(t031_runtime["readiness"]["t031_all_survivors_real_runtime_integrated"])
        self.assertTrue(t031_runtime["candidate"]["measured_runtime_claim"])
        self.assertGreater(t031_runtime["candidate"]["runtime_comparison"]["measured_tokens_per_second"], 0.0)
        self.assertEqual(
            t031_runtime["candidate"]["runtime_comparison"]["generated_token_ids"],
            result["receipt"]["generated_token_ids"],
        )
        self.assertTrue(t031_runtime["candidate"]["runtime_integration_evidence"]["T031-S01"]["real_runtime_path"])
        self.assertTrue(t031_runtime["candidate"]["runtime_integration_evidence"]["T031-S02"]["real_runtime_path"])
        self.assertTrue(t031_runtime["candidate"]["runtime_integration_evidence"]["T031-S03"]["support_runtime_path"])
        self.assertTrue(t031_runtime["candidate"]["runtime_integration_evidence"]["T031-E04"]["real_runtime_path"])
        self.assertTrue(t031_runtime["candidate"]["runtime_integration_evidence"]["T031-E02"]["support_runtime_path"])
        self.assertTrue(t031_runtime["candidate"]["runtime_integration_evidence"]["T031-E07"]["support_runtime_path"])
        t031_receipt = t031_runtime["runtime"]["receipt"]
        self.assertIn("t031_lm_head_certificate", t031_receipt)
        self.assertTrue(t031_receipt["t031_lm_head_certificate"]["active"])
        self.assertTrue(t031_runtime["readiness"]["t033_greedy_lm_head_output_contract_ready"])
        t033_greedy = t031_runtime["candidate"]["t033_greedy_lm_head_output_contract"]
        self.assertTrue(t033_greedy["ready"])
        self.assertFalse(t033_greedy["full_logits_materialized"])
        self.assertFalse(t033_greedy["full_logits_returned_to_host"])
        t033_descriptor_gate = t031_runtime["candidate"]["t033_descriptor_denominator_gate"]
        self.assertTrue(t031_runtime["readiness"]["t033_descriptor_denominator_gate_ready"])
        self.assertGreater(t033_descriptor_gate["descriptor_count"], 0)
        self.assertIn("zero_group_active_tape", t033_descriptor_gate["gates"])
        self.assertIn("scale_chamber_payload", t033_descriptor_gate["gates"])
        first_descriptor = next(iter(t031_receipt["device_payload_descriptor_table"].values()))
        self.assertIn("t031_structural_payload", first_descriptor)
        self.assertIn("T031-S04", t031_runtime["readiness"]["t031_real_runtime_missing_ids"])
        self.assertIn("t031_survivors_not_integrated_into_real_runtime", t031_runtime["blockers"])


if __name__ == "__main__":
    unittest.main()
