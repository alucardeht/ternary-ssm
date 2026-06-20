import subprocess
import sys
import unittest

from src.constructive_imagination import evaluate_r440_room_update, room_update_passed, train_r440_curriculum


class ConstructiveImaginationRoomUpdateTests(unittest.TestCase):
    def test_room_update_changes_only_corrected_layout(self):
        model, _curriculum, result = train_r440_curriculum(seed=101)
        update = evaluate_r440_room_update(model, seed=6101)

        self.assertTrue(result.passed)
        self.assertTrue(update.passed)
        self.assertTrue(room_update_passed(update.metrics))
        self.assertGreaterEqual(update.metrics["updated_exact_layout"], 0.90)
        self.assertGreaterEqual(update.metrics["intended_object_changed"], 0.90)
        self.assertGreaterEqual(update.metrics["unchanged_objects_preserved"], 0.90)
        self.assertLessEqual(update.metrics["stale_exact_layout"], 0.35)
        self.assertLessEqual(update.metrics["slot_ablation_exact_layout"], 0.35)

    def test_growth_probe_runner_enables_room_update(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "101",
                "--room-update",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_GROWTH_PROBE_ACTIVE", completed.stdout)
        self.assertIn("R440_ROOM_UPDATE_ACTIVE", completed.stdout)
        self.assertIn('"room_update": {', completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
