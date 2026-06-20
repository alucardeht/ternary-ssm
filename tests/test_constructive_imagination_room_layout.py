import subprocess
import sys
import unittest

from src.constructive_imagination import evaluate_r440_room_layout, room_layout_passed, train_r440_curriculum


class ConstructiveImaginationRoomLayoutTests(unittest.TestCase):
    def test_room_layout_reconstructs_controlled_wireframe(self):
        model, _curriculum, result = train_r440_curriculum(seed=101)
        room = evaluate_r440_room_layout(model, seed=5101)

        self.assertTrue(result.passed)
        self.assertTrue(room.passed)
        self.assertTrue(room_layout_passed(room.metrics))
        self.assertGreaterEqual(room.metrics["exact_layout"], 0.90)
        self.assertGreaterEqual(room.metrics["no_overlap"], 0.95)
        self.assertGreaterEqual(room.metrics["avoids_wall"], 0.95)
        self.assertLessEqual(room.metrics["slot_ablation_exact_layout"], 0.35)

    def test_growth_probe_runner_enables_room_layout(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "101",
                "--room-layout",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_GROWTH_PROBE_ACTIVE", completed.stdout)
        self.assertIn("R440_ROOM_LAYOUT_ACTIVE", completed.stdout)
        self.assertIn('"room_layout": {', completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
