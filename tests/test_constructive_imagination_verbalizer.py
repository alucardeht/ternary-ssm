import subprocess
import sys
import unittest

from src.constructive_imagination import (
    R440Curriculum,
    R440TinyVerbalizer,
    r440_decode_tokens,
    r440_verbalizer_passed,
    r440_verbalizer_targets,
    train_r440_curriculum,
    train_r440_tiny_verbalizer,
)


class ConstructiveImaginationVerbalizerTests(unittest.TestCase):
    def test_verbalizer_targets_have_two_surface_forms(self):
        curriculum = R440Curriculum(seed=31, batch_size=4)
        batch = curriculum.make_batch("mixed")
        template0 = batch.operator.new_zeros(4)
        template1 = batch.operator.new_ones(4)

        cell_form = r440_verbalizer_targets(batch.targets["answer"], template0, grid=curriculum.config.grid)
        row_col_form = r440_verbalizer_targets(batch.targets["answer"], template1, grid=curriculum.config.grid)

        self.assertTrue(r440_decode_tokens(cell_form[0]).startswith("cell "))
        self.assertTrue(r440_decode_tokens(row_col_form[0]).startswith("at row "))
        self.assertEqual(R440TinyVerbalizer(curriculum.config)(batch.frame1.new_zeros(4, curriculum.config.cells), template0).token_logits.shape[:2], (4, 5))

    def test_tiny_verbalizer_trains_and_preserves_causal_controls(self):
        model, curriculum, core_result = train_r440_curriculum(seed=606)
        _verbalizer, verbalizer_result = train_r440_tiny_verbalizer(model, curriculum, seed=606)

        self.assertTrue(core_result.passed)
        self.assertTrue(verbalizer_result.passed)
        self.assertTrue(r440_verbalizer_passed(verbalizer_result.metrics))
        self.assertGreaterEqual(verbalizer_result.metrics["text"], 0.90)
        self.assertLessEqual(verbalizer_result.metrics["slot_ablation_text"], 0.35)
        self.assertLessEqual(verbalizer_result.metrics["proposition_shuffle_text"], 0.35)
        self.assertGreater(len(verbalizer_result.examples), 0)

    def test_growth_probe_runner_enables_tiny_verbalizer(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "101",
                "--verbalizer",
                "tiny",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_GROWTH_PROBE_ACTIVE", completed.stdout)
        self.assertIn("R440_TINY_VERBALIZER_ACTIVE", completed.stdout)
        self.assertIn('"verbalizer": {', completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
