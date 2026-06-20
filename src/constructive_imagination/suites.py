"""Default executable gate suite for R439B probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .certificates import no_bypass_delta
from .language import canonical_query_operator, count_red_cubes
from .memory import AttractorMemory
from .state import ConstructiveObject, ConstructiveState, StatePatch, apply_patch
from .vision import (
    SyntheticVisualScene,
    caption_only_hidden_prediction,
    delayed_reveal_prediction,
    write_visual_state,
)


@dataclass(frozen=True)
class GateCheck:
    name: str
    observed_pass: bool
    expected_pass: bool
    detail: str = ""


@dataclass(frozen=True)
class GateSuiteReport:
    checks: tuple[GateCheck, ...]

    @property
    def matched(self) -> bool:
        return all(check.observed_pass == check.expected_pass for check in self.checks)

    @property
    def unexpected(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if check.observed_pass != check.expected_pass)


def _state_patch_check() -> GateCheck:
    obj = ConstructiveObject("cube-a", "cube", (0, 0), {"color": "red"})
    state = ConstructiveState(objects={"cube-a": obj}, debt={"boundary": 1.0})
    patch = StatePatch(
        patch_type="move_object",
        target="cube-a",
        payload=(1, 1),
        debt_key="boundary",
        expected_debt_delta=-0.5,
    )
    updated = apply_patch(state, patch)
    passed = updated.objects["cube-a"].position == (1, 1) and updated.total_debt() < state.total_debt()
    return GateCheck("constructive_patch_positive_control", passed, True)


def _visual_twin_check() -> GateCheck:
    caption = "a red cube is hidden behind a screen"
    left = SyntheticVisualScene(caption, "visible", "hidden", (1, 1), (0, 3))
    right = SyntheticVisualScene(caption, "visible", "hidden", (1, 1), (3, 0))
    left_state = write_visual_state(left)
    right_state = write_visual_state(right)
    visual_differs = delayed_reveal_prediction(left_state, "hidden") != delayed_reveal_prediction(
        right_state,
        "hidden",
    )
    caption_same = caption_only_hidden_prediction(caption) == caption_only_hidden_prediction(caption)
    return GateCheck("same_caption_visual_twin_positive_control", visual_differs and caption_same, True)


def _certificate_check() -> GateCheck:
    proof = no_bypass_delta(
        live_score=1.0,
        ablated_score=0.2,
        component="H_t.hidden_continuation",
        intervention="targeted_ablation",
        overlap_class="O1",
        min_delta=0.5,
    )
    return GateCheck("certificate_intervention_positive_control", proof.accepted, True)


def _memory_check() -> GateCheck:
    memory = AttractorMemory(cells=2, width=3, write_threshold=0.5)
    write = memory.write(np.array([1.0, 0.0, 0.0]), surprise=0.7, conflict=0.0)
    before = memory.recall(np.array([1.0, 0.0, 0.0]))
    memory.zero_cell(write.cell_index)
    after = memory.recall(np.array([1.0, 0.0, 0.0]))
    passed = write.written and before[0] > 0.9 and np.allclose(after, 0.0)
    return GateCheck("fixed_budget_memory_positive_control", passed, True)


def _language_check() -> GateCheck:
    state = ConstructiveState(
        objects={
            "a": ConstructiveObject("a", "cube", (0, 0), {"color": "red"}),
            "b": ConstructiveObject("b", "sphere", (0, 1), {"color": "blue"}),
        },
    )
    answer = count_red_cubes(state)
    ablated = count_red_cubes(ConstructiveState())
    same_operator = canonical_query_operator("left of the cup") == canonical_query_operator(
        "a esquerda do copo",
    )
    passed = (
        not answer.abstained
        and "red" in answer.provenance
        and ablated.abstained
        and same_operator
    )
    return GateCheck("language_provenance_positive_control", passed, True)


def run_default_r439b_gate_suite() -> GateSuiteReport:
    return GateSuiteReport(
        checks=(
            _state_patch_check(),
            _visual_twin_check(),
            _certificate_check(),
            _memory_check(),
            _language_check(),
        ),
    )

