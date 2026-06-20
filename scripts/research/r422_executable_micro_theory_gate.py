#!/usr/bin/env python3
"""Executable micro-theory / local simulator substrate gate.

This bounded R422 gate asks whether a compact executable local-theory object can
answer held-out compositional state queries with less paid material than
ordinary examples or trace-supervised transition material.

It is not production training, not GPU work, and not a speed benchmark. The
micro-theory object is charged for build bytes, per-step execution, and exact
state verification. It passes only if the object beats text/example and trace
controls under the same domain family without acting like a free external
simulator.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_executable_micro_theory_gate.v1"
ACTIONS = ("move", "charge", "pickup", "open", "rest")
QUERIES = ("position", "energy", "has_key", "door_open")


@dataclass(frozen=True)
class Domain:
    domain_id: int
    track_len: int
    start_pos: int
    key_pos: int
    door_pos: int
    charger_pos: int
    move_cost: int
    charge_gain: int
    max_energy: int
    rest_gain: int


@dataclass(frozen=True)
class State:
    pos: int
    energy: int
    has_key: bool
    door_open: bool


@dataclass(frozen=True)
class Example:
    domain_id: int
    actions: tuple[str, ...]
    query: str
    answer: int
    final_state: State
    cost: float


@dataclass
class ArmResult:
    arm: str
    spent_budget: float
    material_items: int
    total_accuracy: float
    long_accuracy: float
    state_check_fraction: float
    paid_cost_per_eval: float
    correct_per_paid: float


def make_domains(count: int, rng: random.Random) -> list[Domain]:
    domains: list[Domain] = []
    for did in range(count):
        track_len = rng.choice([7, 8, 9, 10])
        start_pos = rng.randrange(track_len)
        key_pos = rng.randrange(track_len)
        door_pos = rng.randrange(track_len)
        charger_pos = rng.randrange(track_len)
        move_cost = rng.choice([1, 2])
        charge_gain = rng.choice([2, 3, 4])
        max_energy = rng.choice([5, 6, 7])
        rest_gain = rng.choice([1, 2])
        domains.append(
            Domain(
                domain_id=did,
                track_len=track_len,
                start_pos=start_pos,
                key_pos=key_pos,
                door_pos=door_pos,
                charger_pos=charger_pos,
                move_cost=move_cost,
                charge_gain=charge_gain,
                max_energy=max_energy,
                rest_gain=rest_gain,
            )
        )
    return domains


def initial_state(domain: Domain) -> State:
    return State(pos=domain.start_pos, energy=domain.max_energy // 2 + 1, has_key=False, door_open=False)


def step(domain: Domain, state: State, action: str) -> State:
    pos = state.pos
    energy = state.energy
    has_key = state.has_key
    door_open = state.door_open
    if action == "move":
        if energy >= domain.move_cost:
            pos = (pos + 1) % domain.track_len
            energy -= domain.move_cost
    elif action == "charge":
        if pos == domain.charger_pos:
            energy = min(domain.max_energy, energy + domain.charge_gain)
    elif action == "pickup":
        if pos == domain.key_pos:
            has_key = True
    elif action == "open":
        if pos == domain.door_pos and has_key:
            door_open = True
    elif action == "rest":
        energy = min(domain.max_energy, energy + domain.rest_gain)
    else:
        raise ValueError(f"unknown action: {action}")
    return State(pos=pos, energy=energy, has_key=has_key, door_open=door_open)


def run_state(domain: Domain, actions: tuple[str, ...]) -> State:
    state = initial_state(domain)
    for action in actions:
        state = step(domain, state, action)
    return state


def answer_for(state: State, query: str) -> int:
    if query == "position":
        return state.pos
    if query == "energy":
        return state.energy
    if query == "has_key":
        return int(state.has_key)
    if query == "door_open":
        return int(state.door_open)
    raise ValueError(f"unknown query: {query}")


def random_actions(rng: random.Random, min_len: int, max_len: int) -> tuple[str, ...]:
    return tuple(rng.choice(ACTIONS) for _ in range(rng.randint(min_len, max_len)))


def make_example(domain: Domain, actions: tuple[str, ...], query: str, cost: float) -> Example:
    final_state = run_state(domain, actions)
    return Example(
        domain_id=domain.domain_id,
        actions=actions,
        query=query,
        answer=answer_for(final_state, query),
        final_state=final_state,
        cost=cost,
    )


def build_eval(domains: list[Domain], seed: int, per_domain: int, min_len: int, max_len: int) -> list[Example]:
    rng = random.Random(seed)
    rows: list[Example] = []
    for domain in domains:
        for _ in range(per_domain):
            rows.append(make_example(domain, random_actions(rng, min_len, max_len), rng.choice(QUERIES), 0.0))
    return rows


class ExampleMemoryLearner:
    def __init__(self) -> None:
        self.memory: dict[tuple[int, tuple[str, ...], str], int] = {}
        self.query_majority: dict[tuple[int, str], dict[int, int]] = {}

    def observe(self, ex: Example) -> None:
        self.memory[(ex.domain_id, ex.actions, ex.query)] = ex.answer
        bucket = self.query_majority.setdefault((ex.domain_id, ex.query), {})
        bucket[ex.answer] = bucket.get(ex.answer, 0) + 1

    def predict(self, ex: Example) -> tuple[int | None, State | None]:
        key = (ex.domain_id, ex.actions, ex.query)
        if key in self.memory:
            return self.memory[key], None
        bucket = self.query_majority.get((ex.domain_id, ex.query))
        if bucket:
            return max(bucket.items(), key=lambda item: item[1])[0], None
        return None, None


class TraceLearner:
    def __init__(self, domains: list[Domain]) -> None:
        self.domains = {d.domain_id: d for d in domains}
        self.action_delta: dict[tuple[int, str], dict[tuple[int, int, int, int], int]] = {}
        self.memory: dict[tuple[int, tuple[str, ...], str], int] = {}

    def observe(self, ex: Example) -> None:
        self.memory[(ex.domain_id, ex.actions, ex.query)] = ex.answer
        domain = self.domains[ex.domain_id]
        state = initial_state(domain)
        for action in ex.actions:
            nxt = step(domain, state, action)
            delta = (
                (nxt.pos - state.pos) % domain.track_len,
                nxt.energy - state.energy,
                int(nxt.has_key) - int(state.has_key),
                int(nxt.door_open) - int(state.door_open),
            )
            bucket = self.action_delta.setdefault((domain.domain_id, action), {})
            bucket[delta] = bucket.get(delta, 0) + 1
            state = nxt

    def _step_guess(self, domain: Domain, state: State, action: str) -> State:
        bucket = self.action_delta.get((domain.domain_id, action))
        if not bucket:
            return state
        dpos, denergy, dkey, ddoor = max(bucket.items(), key=lambda item: item[1])[0]
        return State(
            pos=(state.pos + dpos) % domain.track_len,
            energy=max(0, min(domain.max_energy, state.energy + denergy)),
            has_key=bool(max(0, min(1, int(state.has_key) + dkey))),
            door_open=bool(max(0, min(1, int(state.door_open) + ddoor))),
        )

    def predict(self, ex: Example) -> tuple[int | None, State | None]:
        key = (ex.domain_id, ex.actions, ex.query)
        if key in self.memory:
            return self.memory[key], None
        domain = self.domains[ex.domain_id]
        state = initial_state(domain)
        for action in ex.actions:
            state = self._step_guess(domain, state, action)
        return answer_for(state, ex.query), state


class MicroTheoryLearner:
    def __init__(self, domains: list[Domain]) -> None:
        self.domains = {d.domain_id: d for d in domains}
        self.enabled: set[int] = set()

    def add_theory(self, domain: Domain) -> None:
        self.enabled.add(domain.domain_id)

    def predict(self, ex: Example) -> tuple[int | None, State | None]:
        if ex.domain_id not in self.enabled:
            return None, None
        domain = self.domains[ex.domain_id]
        state = run_state(domain, ex.actions)
        return answer_for(state, ex.query), state


def build_material(arm: str, domains: list[Domain], budget: float, seed: int) -> tuple[object, float, int]:
    rng = random.Random(seed)
    spent = 0.0
    items = 0
    if arm == "ordinary_examples":
        learner = ExampleMemoryLearner()
        while spent + 1.0 <= budget:
            domain = rng.choice(domains)
            ex = make_example(domain, random_actions(rng, 1, 7), rng.choice(QUERIES), 1.0)
            learner.observe(ex)
            spent += ex.cost
            items += 1
        return learner, spent, items
    if arm == "trace_supervision":
        learner = TraceLearner(domains)
        while spent + 1.2 <= budget:
            domain = rng.choice(domains)
            ex = make_example(domain, random_actions(rng, 2, 8), rng.choice(QUERIES), 1.2)
            learner.observe(ex)
            spent += ex.cost
            items += 1
        return learner, spent, items
    if arm == "executable_micro_theory":
        learner = MicroTheoryLearner(domains)
        for domain in domains:
            build_cost = 9.5 + 0.35 * domain.track_len
            if spent + build_cost > budget:
                break
            learner.add_theory(domain)
            spent += build_cost
            items += 1
        return learner, spent, items
    raise ValueError(f"unknown arm: {arm}")


def evaluate_arm(arm: str, domains: list[Domain], args: argparse.Namespace, trial_seed: int) -> ArmResult:
    learner, spent, items = build_material(arm, domains, args.budget, trial_seed)
    eval_rows = build_eval(domains, trial_seed + 50_000, args.eval_per_domain, args.eval_min_len, args.eval_max_len)
    correct = 0
    long_correct = 0
    long_total = 0
    state_checks = 0
    state_ok = 0
    exec_cost = 0.0
    for ex in eval_rows:
        pred, pred_state = learner.predict(ex)  # type: ignore[attr-defined]
        ok = pred == ex.answer
        correct += int(ok)
        if len(ex.actions) >= args.long_len:
            long_total += 1
            long_correct += int(ok)
        if pred_state is not None:
            state_checks += 1
            state_ok += int(pred_state == ex.final_state)
        if arm == "executable_micro_theory":
            exec_cost += args.exec_cost_per_step * len(ex.actions) + args.verify_cost_per_query
        elif arm == "trace_supervision":
            exec_cost += args.trace_cost_per_step * len(ex.actions) + args.trace_query_cost
        else:
            exec_cost += args.ordinary_cost_per_step * len(ex.actions) + args.ordinary_query_cost
    total = len(eval_rows)
    total_accuracy = correct / total
    long_accuracy = long_correct / max(1, long_total)
    state_check_fraction = state_ok / max(1, state_checks)
    paid_cost_per_eval = (spent + exec_cost) / total
    correct_per_paid = correct / max(1e-9, spent + exec_cost)
    return ArmResult(
        arm=arm,
        spent_budget=spent,
        material_items=items,
        total_accuracy=total_accuracy,
        long_accuracy=long_accuracy,
        state_check_fraction=state_check_fraction,
        paid_cost_per_eval=paid_cost_per_eval,
        correct_per_paid=correct_per_paid,
    )


def run_trial(args: argparse.Namespace, seed: int) -> dict[str, dict]:
    domains = make_domains(args.domains, random.Random(seed))
    return {
        arm: asdict(evaluate_arm(arm, domains, args, seed + idx * 1000))
        for idx, arm in enumerate(("ordinary_examples", "trace_supervision", "executable_micro_theory"))
    }


def aggregate(trials: list[dict[str, dict]]) -> dict:
    arms = sorted(trials[0].keys())
    out: dict[str, dict[str, float]] = {}
    for arm in arms:
        rows = [trial[arm] for trial in trials]
        out[arm] = {
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in rows[0]
            if metric != "arm"
        }
    micro = out["executable_micro_theory"]
    controls = {arm: out[arm] for arm in arms if arm != "executable_micro_theory"}
    best_control_name, best_control = max(controls.items(), key=lambda item: item[1]["total_accuracy"])
    total_gain = micro["total_accuracy"] - best_control["total_accuracy"]
    long_gain = micro["long_accuracy"] - best_control["long_accuracy"]
    correct_per_paid_ratio = micro["correct_per_paid"] / max(best_control["correct_per_paid"], 1e-12)
    state_check_fraction = micro["state_check_fraction"]
    pass_gate = (
        total_gain >= 0.10
        and long_gain >= 0.15
        and correct_per_paid_ratio >= 1.20
        and state_check_fraction >= 0.999
    )
    if pass_gate:
        decision = "PASS_EXECUTABLE_MICRO_THEORY_GATE"
    elif state_check_fraction < 0.999:
        decision = "FAIL_STATE_CHECKS"
    elif total_gain < 0.10:
        decision = "FAIL_TOTAL_CAPABILITY_GAIN"
    elif long_gain < 0.15:
        decision = "FAIL_COMPOSITIONAL_LONG_GAIN"
    else:
        decision = "FAIL_PAID_COST_RATIO"
    return {
        "arms": out,
        "best_control": best_control_name,
        "decision": decision,
        "pass_gate": pass_gate,
        "total_gain": total_gain,
        "long_gain": long_gain,
        "correct_per_paid_ratio": correct_per_paid_ratio,
        "state_check_fraction": state_check_fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", type=int, default=32)
    parser.add_argument("--budget", type=float, default=480.0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--eval-per-domain", type=int, default=40)
    parser.add_argument("--eval-min-len", type=int, default=8)
    parser.add_argument("--eval-max-len", type=int, default=18)
    parser.add_argument("--long-len", type=int, default=13)
    parser.add_argument("--exec-cost-per-step", type=float, default=0.035)
    parser.add_argument("--verify-cost-per-query", type=float, default=0.02)
    parser.add_argument("--trace-cost-per-step", type=float, default=0.07)
    parser.add_argument("--trace-query-cost", type=float, default=0.03)
    parser.add_argument("--ordinary-cost-per-step", type=float, default=0.10)
    parser.add_argument("--ordinary-query-cost", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42291)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.domains < 4:
        raise ValueError("--domains must be at least 4")
    if args.eval_min_len < 1 or args.eval_max_len < args.eval_min_len:
        raise ValueError("invalid eval length range")
    trials = [run_trial(args, args.seed + idx) for idx in range(args.trials)]
    result = aggregate(trials)
    params = vars(args).copy()
    params["output"] = str(args.output) if args.output else None
    payload = {
        "schema": SCHEMA,
        "active": 1,
        "mode": "cpu_executable_micro_theory_gate",
        "no_training_integration": 1,
        "no_gpu": 1,
        "params": params,
        "result": result,
        "trials": trials,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[R422_EXECUTABLE_MICRO_THEORY_GATE] "
        f"active=1 decision={result['decision']} pass={int(result['pass_gate'])} "
        f"best_control={result['best_control']} "
        f"total_gain={result['total_gain']:.6f} "
        f"long_gain={result['long_gain']:.6f} "
        f"correct_per_paid_ratio={result['correct_per_paid_ratio']:.6f} "
        f"state_check_fraction={result['state_check_fraction']:.6f} "
        "no_training_integration=1 no_gpu=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
