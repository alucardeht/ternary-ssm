import unittest

import numpy as np

from src.constructive_imagination import (
    AttractorMemory,
    ConstructiveObject,
    ConstructiveState,
    DenominatorReceipt,
    StatePatch,
    SyntheticVisualScene,
    apply_patch,
    bind_handle,
    canonical_query_operator,
    caption_only_hidden_prediction,
    count_red_cubes,
    delayed_reveal_prediction,
    no_bypass_delta,
    resolve_handle,
    run_default_r439b_gate_suite,
    write_visual_state,
)


class ConstructiveImaginationGateTests(unittest.TestCase):
    def test_typed_patch_changes_target_and_reduces_debt(self):
        state = ConstructiveState(
            objects={
                "cube-a": ConstructiveObject("cube-a", "cube", (0, 0), {"color": "red"}),
                "cube-b": ConstructiveObject("cube-b", "cube", (2, 2), {"color": "blue"}),
            },
            debt={"boundary": 1.0},
        )
        patch = StatePatch(
            patch_type="move_object",
            target="cube-a",
            payload=(1, 1),
            debt_key="boundary",
            expected_debt_delta=-0.75,
        )

        updated = apply_patch(state, patch)

        self.assertEqual(updated.objects["cube-a"].position, (1, 1))
        self.assertEqual(updated.objects["cube-b"].position, (2, 2))
        self.assertLess(updated.total_debt(), state.total_debt())

    def test_same_caption_visual_twins_require_state_not_caption(self):
        caption = "a red cube is hidden behind a screen"
        left = write_visual_state(
            SyntheticVisualScene(caption, "visible", "hidden", (1, 1), (0, 3)),
        )
        right = write_visual_state(
            SyntheticVisualScene(caption, "visible", "hidden", (1, 1), (3, 0)),
        )

        self.assertEqual(caption_only_hidden_prediction(caption), (0, 0))
        self.assertNotEqual(
            delayed_reveal_prediction(left, "hidden"),
            delayed_reveal_prediction(right, "hidden"),
        )

        ablated = ConstructiveState(objects=left.objects, boundary=left.boundary, void=left.void)
        self.assertIsNone(delayed_reveal_prediction(ablated, "hidden"))

    def test_certificate_records_intervention_delta_and_denominator(self):
        proof = no_bypass_delta(
            live_score=0.95,
            ablated_score=0.10,
            component="H_t.hidden_continuation",
            intervention="zero_hidden_state",
            overlap_class="O1",
            min_delta=0.5,
        )
        receipt = DenominatorReceipt(
            progress_gain=1.0,
            build_cost=0.1,
            verification_cost=0.1,
            memory_cost=0.2,
            overlap_class="O1",
        )

        self.assertTrue(proof.accepted)
        self.assertEqual(proof.component, "H_t.hidden_continuation")
        self.assertGreater(proof.delta, 0.5)
        self.assertGreater(receipt.net, 0.0)
        self.assertEqual(receipt.overlap_class, "O1")

    def test_fixed_budget_memory_targeted_ablation(self):
        memory = AttractorMemory(cells=2, width=3, write_threshold=0.5)
        write_a = memory.write(np.array([1.0, 0.0, 0.0]), surprise=0.8, conflict=0.0)
        write_b = memory.write(np.array([0.0, 1.0, 0.0]), surprise=0.8, conflict=0.0)

        recall_a = memory.recall(np.array([1.0, 0.0, 0.0]))
        recall_b = memory.recall(np.array([0.0, 1.0, 0.0]))
        memory.zero_cell(write_a.cell_index)
        recall_a_ablated = memory.recall(np.array([1.0, 0.0, 0.0]))
        recall_b_after = memory.recall(np.array([0.0, 1.0, 0.0]))

        self.assertTrue(write_a.written)
        self.assertTrue(write_b.written)
        self.assertGreater(recall_a[0], 0.9)
        self.assertGreater(recall_b[1], 0.9)
        self.assertLess(recall_a_ablated[0], 0.1)
        self.assertGreater(recall_b_after[1], 0.9)

    def test_language_interface_requires_provenance_and_handles(self):
        state = ConstructiveState(
            objects={
                "cube-a": ConstructiveObject("cube-a", "cube", (0, 0), {"color": "red"}),
                "cube-b": ConstructiveObject("cube-b", "cube", (1, 0), {"color": "blue"}),
            },
        )
        answer = count_red_cubes(state)
        ablated = count_red_cubes(ConstructiveState())
        handled = bind_handle(state, "touched-before", "cube-a", event="touch")

        self.assertFalse(answer.abstained)
        self.assertEqual(answer.provenance["red"], "O_t.attributes.color")
        self.assertTrue(ablated.abstained)
        self.assertEqual(resolve_handle(handled, "touched-before"), "cube-a")
        self.assertEqual(
            canonical_query_operator("put the cube left of the sphere"),
            canonical_query_operator("coloque o cubo a esquerda da esfera"),
        )

    def test_default_r439b_gate_suite_matches_controls(self):
        suite = run_default_r439b_gate_suite()

        self.assertTrue(suite.matched)
        self.assertEqual(len(suite.unexpected), 0)
        self.assertEqual(len(suite.checks), 5)


if __name__ == "__main__":
    unittest.main()

