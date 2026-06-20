import unittest

from src.constructive_imagination import run_all_trainable_probes, run_r440_slot_proposition_depth_probe


class ConstructiveImaginationTrainableTests(unittest.TestCase):
    def test_trainable_probes_pass_with_causal_controls(self):
        results = run_all_trainable_probes(seed=101)

        by_name = {result.name: result for result in results}
        self.assertEqual(
            set(by_name),
            {
                "combined_world",
                "latent_write_world",
                "pixel_latent_world",
                "slot_distractor_world",
                "occlusion_permanence_world",
                "r440_slot_proposition_depth",
                "patch_dynamics",
                "visual_state_write",
                "internal_memory",
                "language_interface",
            },
        )
        self.assertTrue(all(result.passed for result in results))

        self.assertGreaterEqual(by_name["patch_dynamics"].model_metric, 0.95)
        self.assertGreaterEqual(
            by_name["patch_dynamics"].model_metric - by_name["patch_dynamics"].ablation_metric,
            0.45,
        )

        self.assertGreaterEqual(by_name["visual_state_write"].model_metric, 0.95)
        self.assertLessEqual(by_name["visual_state_write"].control_metric, 0.20)
        self.assertLessEqual(by_name["visual_state_write"].ablation_metric, 0.25)

        self.assertGreaterEqual(by_name["internal_memory"].model_metric, 0.95)
        self.assertLessEqual(by_name["internal_memory"].control_metric, 0.25)
        self.assertLessEqual(by_name["internal_memory"].ablation_metric, 0.25)

        self.assertGreaterEqual(by_name["language_interface"].model_metric, 0.90)
        self.assertLessEqual(by_name["language_interface"].control_metric, 0.35)
        self.assertLessEqual(by_name["language_interface"].ablation_metric, 0.35)

        combined = by_name["combined_world"]
        self.assertGreaterEqual(combined.model_metric, 0.95)
        self.assertLessEqual(combined.control_metric, 0.30)
        self.assertLessEqual(combined.ablation_metric, 0.45)
        self.assertTrue(all(score >= 0.95 for score in combined.task_metrics.values()))

        latent = by_name["latent_write_world"]
        self.assertGreaterEqual(latent.model_metric, 0.92)
        self.assertLessEqual(latent.control_metric, 0.32)
        self.assertLessEqual(latent.ablation_metric, 0.32)
        self.assertGreaterEqual(latent.task_metrics["hidden_location"], 0.90)
        self.assertGreaterEqual(latent.task_metrics["next_pos"], 0.90)
        self.assertGreaterEqual(latent.task_metrics["memory_key"], 0.90)
        self.assertGreaterEqual(latent.task_metrics["red_count"], 0.90)

        pixel = by_name["pixel_latent_world"]
        self.assertGreaterEqual(pixel.model_metric, 0.90)
        self.assertLessEqual(pixel.control_metric, 0.32)
        self.assertLessEqual(pixel.ablation_metric, 0.32)
        self.assertGreaterEqual(pixel.task_metrics["hidden_location"], 0.85)
        self.assertGreaterEqual(pixel.task_metrics["next_pos"], 0.85)
        self.assertGreaterEqual(pixel.task_metrics["memory_key"], 0.85)
        self.assertGreaterEqual(pixel.task_metrics["red_count"], 0.85)

        slot = by_name["slot_distractor_world"]
        self.assertGreaterEqual(slot.model_metric, 0.90)
        self.assertLessEqual(slot.control_metric, 0.28)
        self.assertLessEqual(slot.ablation_metric, 0.30)
        self.assertGreaterEqual(slot.task_metrics["target_position"], 0.85)
        self.assertGreaterEqual(slot.task_metrics["target_next_pos"], 0.85)
        self.assertGreaterEqual(slot.task_metrics["memory_key"], 0.85)
        self.assertLessEqual(slot.task_ablation_metrics["query_swap_target_position"], 0.35)

        occlusion = by_name["occlusion_permanence_world"]
        self.assertGreaterEqual(occlusion.model_metric, 0.90)
        self.assertLessEqual(occlusion.control_metric, 0.36)
        self.assertLessEqual(occlusion.ablation_metric, 0.34)
        self.assertGreaterEqual(occlusion.task_metrics["remembered_position"], 0.85)
        self.assertGreaterEqual(occlusion.task_metrics["remembered_next_pos"], 0.85)
        self.assertGreaterEqual(occlusion.task_metrics["occluded_color"], 0.85)
        self.assertLessEqual(occlusion.task_ablation_metrics["query_swap_remembered_position"], 0.35)

        r440 = by_name["r440_slot_proposition_depth"]
        self.assertGreaterEqual(r440.model_metric, 0.88)
        self.assertLessEqual(r440.control_metric, 0.25)
        self.assertLessEqual(r440.ablation_metric, 0.35)
        self.assertGreaterEqual(r440.task_metrics["answer"], 0.90)
        self.assertGreaterEqual(r440.task_metrics["depth_gate"], 0.90)
        self.assertGreaterEqual(r440.task_metrics["fast_easy_answer"], 0.85)
        self.assertGreaterEqual(r440.task_metrics["deep_hard_answer"], 0.85)
        self.assertGreaterEqual(r440.task_metrics["depth_separation"], 0.50)
        self.assertLessEqual(r440.task_ablation_metrics["query_swap_original_answer"], 0.35)
        self.assertLessEqual(r440.task_ablation_metrics["proposition_shuffle_original_answer"], 0.35)
        self.assertLessEqual(r440.task_ablation_metrics["force_fast_hard_answer"], 0.35)

    def test_r440_probe_passes_with_targeted_controls(self):
        result = run_r440_slot_proposition_depth_probe(seed=107)

        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.task_metrics["answer"], 0.90)
        self.assertGreaterEqual(result.task_metrics["depth_gate"], 0.90)
        self.assertGreaterEqual(result.task_metrics["depth_separation"], 0.50)
        self.assertLessEqual(result.task_ablation_metrics["query_swap_original_answer"], 0.35)
        self.assertLessEqual(result.task_ablation_metrics["proposition_shuffle_original_answer"], 0.35)
        self.assertLessEqual(result.task_ablation_metrics["force_fast_hard_answer"], 0.35)


if __name__ == "__main__":
    unittest.main()
