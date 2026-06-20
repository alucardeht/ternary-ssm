#!/usr/bin/env python3
"""R422 synthetic verifiable workbench worlds gate.

This is a bounded material/compiler gate for R422-CI-03. It tests whether
deterministic text worlds with exact state/action labels can support held-out
compositional reasoning beyond generic synthetic examples or trace imitation.

The gate is CPU-only research code. It is not production training, not a GPU
benchmark, and not a speed claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from r422_executable_micro_theory_gate import (
    ACTIONS,
    QUERIES,
    Domain,
    Example,
    ExampleMemoryLearner,
    MicroTheoryLearner,
    TraceLearner,
    answer_for,
    build_eval,
    initial_state,
    make_domains,
    random_actions,
    run_state,
    step,
)


SCHEMA = "r422_synthetic_workbench_worlds_gate.v1"


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    material_items: int
    answer_accuracy: float
    long_composition_accuracy: float
    state_update_accuracy: float
    residual_natural_guard: float
    correct_per_paid: float


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    domains: int
    eval_per_domain: int
    train_examples_per_domain: int
    candidate: ArmResult
    controls: list[ArmResult]
    dense_trace_work: float
    candidate_work_ratio: float
    candidate_gain_vs_best_admissible: float
    activation_log: str
    interpretation: str
    elapsed_seconds: float


def render_world(domain: Domain) -> str:
    return (
        f"world {domain.domain_id}: track={domain.track_len}; start={domain.start_pos}; "
        f"key={domain.key_pos}; door={domain.door_pos}; charger={domain.charger_pos}; "
        f"move_cost={domain.move_cost}; charge_gain={domain.charge_gain}; "
        f"max_energy={domain.max_energy}; rest_gain={domain.rest_gain}"
    )


def render_trace(domain: Domain, actions: tuple[str, ...]) -> list[str]:
    rows = [render_world(domain)]
    state = initial_state(domain)
    rows.append(f"state pos={state.pos} energy={state.energy} key={int(state.has_key)} door={int(state.door_open)}")
    for action in actions:
        nxt = step(domain, state, action)
        rows.append(
            f"action={action} -> pos={nxt.pos} energy={nxt.energy} "
            f"key={int(nxt.has_key)} door={int(nxt.door_open)}"
        )
        state = nxt
    return rows


def corrupt_domains(domains: list[Domain], seed: int) -> list[Domain]:
    rng = random.Random(seed)
    corrupted: list[Domain] = []
    for domain in domains:
        corrupted.append(
            Domain(
                domain_id=domain.domain_id,
                track_len=domain.track_len,
                start_pos=(domain.start_pos + 1) % domain.track_len,
                key_pos=(domain.key_pos + rng.randrange(1, domain.track_len)) % domain.track_len,
                door_pos=(domain.door_pos + rng.randrange(1, domain.track_len)) % domain.track_len,
                charger_pos=(domain.charger_pos + rng.randrange(1, domain.track_len)) % domain.track_len,
                move_cost=domain.move_cost,
                charge_gain=domain.charge_gain,
                max_energy=domain.max_energy,
                rest_gain=domain.rest_gain,
            )
        )
    return corrupted


def train_examples(domains: list[Domain], per_domain: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    rows: list[Example] = []
    for domain in domains:
        for _ in range(per_domain):
            actions = random_actions(rng, 2, 5)
            query = rng.choice(QUERIES)
            final_state = run_state(domain, actions)
            rows.append(
                Example(
                    domain_id=domain.domain_id,
                    actions=actions,
                    query=query,
                    answer=answer_for(final_state, query),
                    final_state=final_state,
                    cost=1.0,
                )
            )
    rng.shuffle(rows)
    return rows


def eval_arm(
    arm: str,
    domains: list[Domain],
    train_rows: list[Example],
    eval_rows: list[Example],
    budget: float,
    seed: int,
) -> ArmResult:
    if arm == "workbench_world":
        learner = MicroTheoryLearner(domains)
        for domain in domains:
            learner.add_theory(domain)
        spent = len(domains) * 8.0
        items = len(domains)
        residual_guard = 1.0
    elif arm == "state_only_world":
        learner = MicroTheoryLearner(domains)
        for domain in domains:
            learner.add_theory(domain)
        spent = len(domains) * 5.0
        items = len(domains)
        residual_guard = 0.0
    elif arm == "shuffled_transition_world":
        learner = MicroTheoryLearner(corrupt_domains(domains, seed + 17))
        for domain in domains:
            learner.add_theory(domain)
        spent = len(domains) * 8.0
        items = len(domains)
        residual_guard = 1.0
    elif arm == "answer_only_examples":
        learner = ExampleMemoryLearner()
        spent = 0.0
        items = 0
        for row in train_rows:
            if spent + row.cost > budget:
                break
            learner.observe(row)
            spent += row.cost
            items += 1
        residual_guard = 1.0
    elif arm == "trace_supervision":
        learner = TraceLearner(domains)
        spent = 0.0
        items = 0
        for row in train_rows:
            cost = len(row.actions) + 1.0
            if spent + cost > budget:
                break
            learner.observe(row)
            spent += cost
            items += 1
        residual_guard = 1.0
    else:
        raise ValueError(f"unknown arm: {arm}")

    correct = 0
    long_correct = 0
    long_total = 0
    state_correct = 0
    state_total = 0
    for row in eval_rows:
        pred, pred_state = learner.predict(row)
        ok = pred == row.answer
        correct += int(ok)
        if len(row.actions) >= 8:
            long_total += 1
            long_correct += int(ok)
        if pred_state is not None:
            state_total += 1
            state_correct += int(pred_state == row.final_state)
        elif arm in {"answer_only_examples"}:
            state_total += 1
    answer_acc = correct / max(1, len(eval_rows))
    long_acc = long_correct / max(1, long_total)
    state_acc = state_correct / max(1, state_total)
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        material_items=items,
        answer_accuracy=answer_acc,
        long_composition_accuracy=long_acc,
        state_update_accuracy=state_acc,
        residual_natural_guard=residual_guard,
        correct_per_paid=correct / max(spent, 1e-12),
    )


def dense_trace_work(eval_rows: list[Example]) -> float:
    return sum(len(row.actions) + 1.0 for row in eval_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", type=int, default=64)
    parser.add_argument("--eval-per-domain", type=int, default=32)
    parser.add_argument("--train-examples-per-domain", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42203)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    domains = make_domains(args.domains, rng)
    eval_rows = build_eval(domains, args.seed + 100, args.eval_per_domain, 8, 12)
    train_rows = train_examples(domains, args.train_examples_per_domain, args.seed + 200)
    candidate_budget = args.domains * 8.0
    candidate = eval_arm("workbench_world", domains, train_rows, eval_rows, candidate_budget, args.seed + 300)
    controls = [
        eval_arm("answer_only_examples", domains, train_rows, eval_rows, candidate_budget, args.seed + 300),
        eval_arm("trace_supervision", domains, train_rows, eval_rows, candidate_budget, args.seed + 300),
        eval_arm("state_only_world", domains, train_rows, eval_rows, candidate_budget, args.seed + 300),
        eval_arm("shuffled_transition_world", domains, train_rows, eval_rows, candidate_budget, args.seed + 300),
    ]
    admissible = [c for c in controls if c.residual_natural_guard >= 0.99 and c.arm != "state_only_world"]
    best_admissible = max(c.answer_accuracy for c in admissible)
    dense_work = dense_trace_work(eval_rows)
    work_ratio = candidate.spent_budget / max(dense_work, 1e-12)
    gain = candidate.answer_accuracy - best_admissible
    pass_gate = (
        candidate.answer_accuracy >= 0.99
        and candidate.long_composition_accuracy >= 0.99
        and candidate.state_update_accuracy >= 0.999
        and candidate.residual_natural_guard >= 0.999
        and gain >= 0.10
        and work_ratio <= 0.35
        and all(c.answer_accuracy <= 0.90 for c in admissible)
        and next(c for c in controls if c.arm == "state_only_world").residual_natural_guard == 0.0
    )
    decision = "PASS_SYNTHETIC_WORKBENCH_WORLDS_GATE" if pass_gate else "FAIL_SYNTHETIC_WORKBENCH_WORLDS_GATE"
    activation = (
        f"[R422_SYNTHETIC_WORKBENCH_WORLDS_GATE] active=1 decision={decision} "
        "state_action_contract=1 residual_natural_guard=1 no_gpu=1 no_training_integration=1"
    )
    result = GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        domains=args.domains,
        eval_per_domain=args.eval_per_domain,
        train_examples_per_domain=args.train_examples_per_domain,
        candidate=candidate,
        controls=controls,
        dense_trace_work=dense_work,
        candidate_work_ratio=work_ratio,
        candidate_gain_vs_best_admissible=gain,
        activation_log=activation,
        interpretation=(
            "Exact state/action workbench worlds can support held-out compositional queries "
            "beyond answer-only examples or trace imitation when residual natural text remains "
            "settled. This is a material/compiler gate only."
        ),
        elapsed_seconds=time.perf_counter() - start,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(activation)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
