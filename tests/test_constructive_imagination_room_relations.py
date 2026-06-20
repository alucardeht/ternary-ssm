import subprocess
import sys
import unittest

from src.constructive_imagination import evaluate_r440_room_relations, room_relation_passed, train_r440_curriculum


class ConstructiveImaginationRoomRelationsTests(unittest.TestCase):
    def test_room_relations_match_reconstructed_layout(self):
        model, _curriculum, result = train_r440_curriculum(seed=101)
        relations = evaluate_r440_room_relations(model, seed=7101)

        self.assertTrue(result.passed)
        self.assertTrue(relations.passed)
        self.assertTrue(room_relation_passed(relations.metrics))
        self.assertGreaterEqual(relations.metrics["relation_accuracy"], 0.95)
        self.assertGreaterEqual(relations.metrics["exact_relation_set"], 0.90)
        self.assertLessEqual(relations.metrics["slot_ablation_exact_relation_set"], 0.35)

    def test_growth_probe_runner_enables_room_relations(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/research/r440_growth_probe.py",
                "--seed",
                "101",
                "--room-relations",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("R440_GROWTH_PROBE_ACTIVE", completed.stdout)
        self.assertIn("R440_ROOM_RELATIONS_ACTIVE", completed.stdout)
        self.assertIn('"room_relations": {', completed.stdout)
        self.assertIn('"passed": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
