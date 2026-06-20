import subprocess
import sys
import unittest

from src.constructive_imagination import r440_appearance_object_files_passed, train_r440_appearance_object_files


class ConstructiveImaginationAppearanceFilesTests(unittest.TestCase):
    def test_appearance_object_files_decode_identity_shape_position_and_presence(self):
        _bridge, _renderer, result = train_r440_appearance_object_files(seed=101, batch_size=64)

        self.assertTrue(result.passed)
        self.assertTrue(r440_appearance_object_files_passed(result.metrics))
        self.assertGreaterEqual(result.metrics["identity_accuracy"], 0.95)
        self.assertGreaterEqual(result.metrics["shape_accuracy"], 0.95)
        self.assertGreaterEqual(result.metrics["position_accuracy"], 0.95)
        self.assertGreaterEqual(result.metrics["exact_object_files"], 0.90)
        self.assertGreaterEqual(result.metrics["translated_exact_object_files"], 0.90)
        self.assertGreaterEqual(result.metrics["delete_exact_object_files"], 0.90)
        self.assertLessEqual(result.metrics["blank_presence_rate"], 0.20)
        self.assertLessEqual(result.metrics["identity_ablation_exact_object_files"], 0.35)
        self.assertLessEqual(result.metrics["identity_ablation_identity_accuracy"], 0.60)
        self.assertLessEqual(result.metrics["shape_ablation_exact_object_files"], 0.35)
        self.assertLessEqual(result.metrics["shape_ablation_shape_accuracy"], 0.60)

    def test_growth_probe_runner_enables_appearance_object_files(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "101",
                "--appearance-object-files",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_APPEARANCE_OBJECT_FILES_ACTIVE", completed.stdout)
        self.assertIn('"appearance_object_files": {', completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
