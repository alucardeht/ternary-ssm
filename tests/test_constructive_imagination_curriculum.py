import subprocess
import sys
import unittest

from src.constructive_imagination import (
    R440_CURRICULUM_V1,
    R440Curriculum,
    evaluate_r440_core,
    evaluate_r440_seed_grid,
    stage_passed,
    train_r440_curriculum,
)


class ConstructiveImaginationCurriculumTests(unittest.TestCase):
    def test_curriculum_generates_named_stage_batches(self):
        curriculum = R440Curriculum(seed=17, batch_size=16)

        for stage in R440_CURRICULUM_V1:
            batch = curriculum.make_batch(stage)
            self.assertEqual(batch.frame1.shape, (16, curriculum.config.cells, curriculum.config.frame_dim))
            self.assertEqual(batch.frame2.shape, (16, curriculum.config.cells, curriculum.config.frame_dim))
            self.assertEqual(batch.extra.shape, (16, curriculum.config.extra_dim))
            self.assertEqual(batch.targets["answer"].shape, (16,))
            self.assertEqual(batch.stage, stage)

        self.assertFalse(curriculum.make_batch("visible_position").hard.any())
        self.assertTrue(curriculum.make_batch("occluded_transition").hard.all())

    def test_curriculum_v1_trains_and_passes_controls(self):
        model, curriculum, result = train_r440_curriculum(seed=909)

        self.assertTrue(result.passed)
        self.assertEqual(tuple(result.stages), R440_CURRICULUM_V1)
        self.assertGreater(result.cost_profile["total_params"], 0)
        for stage in R440_CURRICULUM_V1:
            self.assertTrue(stage_passed(stage, result.metrics[stage]))

        heldout = evaluate_r440_core(model, R440Curriculum(seed=1909, batch_size=curriculum.batch_size), stage="mixed")
        self.assertTrue(stage_passed("mixed", heldout))

    def test_seed_grid_preserves_fast_path_on_known_regression_seed(self):
        result = evaluate_r440_seed_grid(seeds=(101, 808))

        self.assertTrue(result.passed)
        self.assertEqual(result.seeds, (101, 808))
        for seed_result in result.results.values():
            self.assertGreaterEqual(seed_result.metrics["visible_position"]["fast_easy_answer"], 0.85)
            self.assertGreaterEqual(seed_result.metrics["visible_transition"]["fast_easy_answer"], 0.85)
            self.assertGreaterEqual(seed_result.metrics["mixed"]["fast_easy_answer"], 0.85)

    def test_growth_probe_runner_emits_activation_line(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "707",
                "--steps-per-stage",
                "320",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_GROWTH_PROBE_ACTIVE", completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
