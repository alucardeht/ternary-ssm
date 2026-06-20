import subprocess
import sys
import unittest

from src.constructive_imagination import r440_sensory_bridge_passed, train_r440_sensory_bridge


class ConstructiveImaginationSensoryBridgeTests(unittest.TestCase):
    def test_sensory_bridge_builds_object_state_from_image_tensor(self):
        _bridge, _renderer, result = train_r440_sensory_bridge(seed=101, batch_size=32)

        self.assertTrue(result.passed)
        self.assertTrue(r440_sensory_bridge_passed(result.metrics))
        self.assertGreaterEqual(result.metrics["exact_layout"], 0.90)
        self.assertGreaterEqual(result.metrics["translated_exact_layout"], 0.90)
        self.assertGreaterEqual(result.metrics["delete_visible_exact_layout"], 0.90)
        self.assertLessEqual(result.metrics["blank_exact_layout"], 0.35)
        self.assertLessEqual(result.metrics["object_ablation_exact_layout"], 0.35)
        self.assertLessEqual(result.metrics["blank_presence_rate"], 0.35)
        self.assertLessEqual(result.metrics["ablated_presence_rate"], 0.50)

    def test_growth_probe_runner_enables_sensory_bridge(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "101",
                "--sensory-bridge",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_SENSORY_BRIDGE_ACTIVE", completed.stdout)
        self.assertIn('"sensory_bridge": {', completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
