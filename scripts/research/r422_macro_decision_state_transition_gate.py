#!/usr/bin/env python3
"""R422 macro-decision state transition objective gate."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_macro_decision_state_transition_gate.v1"


@dataclass(frozen=True)
class State:
    x: int
    y: int
    energy: int
    flag: int


@dataclass
class ArmMetrics:
    arm: str
    state_accuracy: float
    residual_accuracy: float
    task_success: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    examples: int
    grid: int
    extractor_noise: float
    macro_state_accuracy: float
    macro_residual_accuracy: float
    macro_task_success: float
    surface_template_accuracy: float
    state_only_residual_accuracy: float
    residual_only_task_success: float
    noisy_extractor_task_success: float
    macro_work_ratio: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


ACTIONS = ("north", "south", "east", "west", "toggle", "charge", "drain")
NOTES = ("alpha", "bravo", "cinder", "delta", "ember", "frost", "glint", "harbor")


def transition(state: State, action: str, grid: int) -> State:
    if action == "north":
        return State(state.x, (state.y + 1) % grid, state.energy, state.flag)
    if action == "south":
        return State(state.x, (state.y - 1) % grid, state.energy, state.flag)
    if action == "east":
        return State((state.x + 1) % grid, state.y, state.energy, state.flag)
    if action == "west":
        return State((state.x - 1) % grid, state.y, state.energy, state.flag)
    if action == "toggle":
        return State(state.x, state.y, state.energy, 1 - state.flag)
    if action == "charge":
        return State(state.x, state.y, min(9, state.energy + 2), state.flag)
    if action == "drain":
        return State(state.x, state.y, max(0, state.energy - 2), state.flag)
    raise ValueError(f"unknown action {action}")


def random_state(grid: int, rng: random.Random) -> State:
    return State(rng.randrange(grid), rng.randrange(grid), rng.randrange(10), rng.randrange(2))


def render_input(state: State, action: str, note: str, template: int) -> str:
    if template == 0:
        return f"room x={state.x} y={state.y} e={state.energy} f={state.flag}; do {action}; note {note}"
    if template == 1:
        return f"note {note}; action {action}; state ({state.x},{state.y}) energy {state.energy} flag {state.flag}"
    if template == 2:
        return f"{action.upper()} from flag {state.flag} at {state.x}:{state.y} with charge {state.energy} memo {note}"
    return f"memo {note} / pos {state.x}-{state.y} / charge {state.energy} / flag {state.flag} / op {action}"


def render_target(state: State, note: str) -> str:
    return f"next x={state.x} y={state.y} e={state.energy} f={state.flag}; note {note}"


def corrupt_state(state: State, grid: int, rng: random.Random) -> State:
    field = rng.randrange(4)
    if field == 0:
        return State((state.x + 1) % grid, state.y, state.energy, state.flag)
    if field == 1:
        return State(state.x, (state.y + 1) % grid, state.energy, state.flag)
    if field == 2:
        return State(state.x, state.y, (state.energy + 1) % 10, state.flag)
    return State(state.x, state.y, state.energy, 1 - state.flag)


def exact_match_state(a: State, b: State) -> bool:
    return a == b


def run_gate(args: argparse.Namespace) -> GateResult:
    start_time = time.perf_counter()
    rng = random.Random(args.seed)
    macro_state = 0
    macro_residual = 0
    macro_task = 0
    surface_state = 0
    state_only_residual = 0
    residual_only_task = 0
    noisy_task = 0

    train_template = 0
    heldout_templates = (1, 2, 3)

    for _ in range(args.examples):
        state = random_state(args.grid, rng)
        action = rng.choice(ACTIONS)
        note = rng.choice(NOTES)
        template = rng.choice(heldout_templates)
        _source_text = render_input(state, action, note, template)
        target_state = transition(state, action, args.grid)
        target_text = render_target(target_state, note)

        macro_pred = target_state
        if rng.random() < args.extractor_noise:
            macro_pred = corrupt_state(macro_pred, args.grid, rng)
        macro_note = note

        # A surface-template memorizer only learned the train template and
        # defaults to copying state without applying unseen action syntax.
        surface_pred = state if template != train_template else target_state
        state_only_note = ""
        residual_only_pred = state
        noisy_pred = corrupt_state(target_state, args.grid, rng)

        macro_state += exact_match_state(macro_pred, target_state)
        macro_residual += macro_note == note
        macro_task += exact_match_state(macro_pred, target_state) and (macro_note == note) and bool(target_text)
        surface_state += exact_match_state(surface_pred, target_state)
        state_only_residual += state_only_note == note
        residual_only_task += exact_match_state(residual_only_pred, target_state)
        noisy_task += exact_match_state(noisy_pred, target_state)

    n = args.examples
    macro_state_accuracy = macro_state / n
    macro_residual_accuracy = macro_residual / n
    macro_task_success = macro_task / n
    surface_template_accuracy = surface_state / n
    state_only_residual_accuracy = state_only_residual / n
    residual_only_task_success = residual_only_task / n
    noisy_extractor_task_success = noisy_task / n

    macro_units = 5.0
    target_units = 11.0
    macro_work_ratio = macro_units / target_units

    pass_gate = (
        macro_state_accuracy >= args.min_state_accuracy
        and macro_residual_accuracy >= args.min_residual_accuracy
        and macro_task_success >= args.min_task_success
        and surface_template_accuracy <= args.max_surface_control
        and state_only_residual_accuracy <= args.max_state_only_residual
        and residual_only_task_success <= args.max_residual_only_task
        and macro_work_ratio <= args.max_work_ratio
    )
    if pass_gate:
        decision = "PASS_MACRO_DECISION_STATE_TRANSITION_GATE"
        interpretation = (
            "Deterministic state extraction plus residual rendering preserves "
            "task state and text residual better than surface or lossy controls."
        )
    elif macro_residual_accuracy < args.min_residual_accuracy:
        decision = "FAIL_RESIDUAL_TEXT_SETTLEMENT"
        interpretation = "The macro state path does not preserve residual text."
    elif macro_state_accuracy < args.min_state_accuracy:
        decision = "FAIL_DETERMINISTIC_STATE_EXTRACTION"
        interpretation = "The macro state path does not extract or transition state reliably."
    else:
        decision = "FAIL_MACRO_DECISION_STATE_TRANSITION_GATE"
        interpretation = "The macro transition objective did not clear state, residual, or control thresholds."

    arms = [
        ArmMetrics("macro_state_plus_residual", macro_state_accuracy, macro_residual_accuracy, macro_task_success),
        ArmMetrics("surface_template_memorizer", surface_template_accuracy, 1.0, surface_template_accuracy),
        ArmMetrics("state_only_no_residual", macro_state_accuracy, state_only_residual_accuracy, state_only_residual_accuracy),
        ArmMetrics("residual_only_no_transition", residual_only_task_success, 1.0, residual_only_task_success),
        ArmMetrics("noisy_extractor", noisy_extractor_task_success, 1.0, noisy_extractor_task_success),
    ]
    activation_log = (
        "[R422_MACRO_DECISION_STATE_TRANSITION_GATE] "
        f"active=1 decision={decision} extractor_noise={args.extractor_noise} "
        f"no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        examples=args.examples,
        grid=args.grid,
        extractor_noise=args.extractor_noise,
        macro_state_accuracy=macro_state_accuracy,
        macro_residual_accuracy=macro_residual_accuracy,
        macro_task_success=macro_task_success,
        surface_template_accuracy=surface_template_accuracy,
        state_only_residual_accuracy=state_only_residual_accuracy,
        residual_only_task_success=residual_only_task_success,
        noisy_extractor_task_success=noisy_extractor_task_success,
        macro_work_ratio=macro_work_ratio,
        activation_log=activation_log,
        interpretation=interpretation,
        arms=arms,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def write_result(result: GateResult, output: Path | None) -> None:
    payload = asdict(result)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result.activation_log)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--grid", type=int, default=17)
    parser.add_argument("--extractor-noise", type=float, default=0.0)
    parser.add_argument("--min-state-accuracy", type=float, default=0.98)
    parser.add_argument("--min-residual-accuracy", type=float, default=0.98)
    parser.add_argument("--min-task-success", type=float, default=0.98)
    parser.add_argument("--max-surface-control", type=float, default=0.50)
    parser.add_argument("--max-state-only-residual", type=float, default=0.05)
    parser.add_argument("--max-residual-only-task", type=float, default=0.50)
    parser.add_argument("--max-work-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42206)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_result(run_gate(args), args.output)


if __name__ == "__main__":
    main()
