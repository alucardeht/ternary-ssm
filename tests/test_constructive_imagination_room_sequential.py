import subprocess
import sys
import unittest

from src.constructive_imagination import (
    evaluate_r440_room_sequential_updates,
    room_sequential_update_passed,
    train_r440_curriculum,
)


class ConstructiveImaginationRoomSequentialTests(unittest.TestCase):
    def test_room_sequential_updates_reject_intermediate_state(self):
        model, _curriculum, result = train_r440_curriculum(seed=101)
        sequential = evaluate_r440_room_sequential_updates(model, seed=8101)

        self.assertTrue(result.passed)
        self.assertTrue(sequential.passed)
        self.assertTrue(room_sequential_update_passed(sequential.metrics))
        self.assertGreaterEqual(sequential.metrics["final_exact_layout"], 0.90)
        self.assertGreaterEqual(sequential.metrics["unmoved_object_preserved"], 0.90)
        self.assertLessEqual(sequential.metrics["intermediate_stale_exact_layout"], 0.35)
        self.assertLessEqual(sequential.metrics["slot_ablation_exact_layout"], 0.35)

    def test_growth_probe_runner_enables_room_sequential_updates(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "101",
                "--room-sequential-updates",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_GROWTH_PROBE_ACTIVE", completed.stdout)
        self.assertIn("R440_ROOM_SEQUENTIAL_UPDATES_ACTIVE", completed.stdout)
        self.assertIn('"room_sequential_updates": {', completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
