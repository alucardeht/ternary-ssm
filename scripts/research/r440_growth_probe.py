from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constructive_imagination import (
    R440_CURRICULUM_V1,
    evaluate_r440_seed_grid,
    evaluate_r440_room_layout,
    evaluate_r440_room_relations,
    evaluate_r440_room_sequential_updates,
    evaluate_r440_room_update,
    train_r440_appearance_object_files,
    train_r440_sensory_bridge,
    train_r440_curriculum,
    train_r440_tiny_verbalizer,
    train_r440_variable_object_files,
)


def curriculum_payload(result):
    return {
        "seed": result.seed,
        "steps_per_stage": result.steps_per_stage,
        "stages": list(result.stages),
        "passed": result.passed,
        "metrics": result.metrics,
        "cost_profile": result.cost_profile,
    }


def parse_seed_grid(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("seed grid must include at least one seed")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R440 constructive growth curriculum probe.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps-per-stage", type=int, default=320)
    parser.add_argument("--verbalizer-steps", type=int, default=160)
    parser.add_argument("--curriculum", choices=["v1"], default="v1")
    parser.add_argument("--verbalizer", choices=["none", "tiny"], default="none")
    parser.add_argument("--seed-grid", type=parse_seed_grid, default=None)
    parser.add_argument("--room-layout", action="store_true")
    parser.add_argument("--room-relations", action="store_true")
    parser.add_argument("--room-sequential-updates", action="store_true")
    parser.add_argument("--room-update", action="store_true")
    parser.add_argument("--sensory-bridge", action="store_true")
    parser.add_argument("--sensory-steps", type=int, default=240)
    parser.add_argument("--variable-object-files", action="store_true")
    parser.add_argument("--variable-object-steps", type=int, default=900)
    parser.add_argument("--appearance-object-files", action="store_true")
    parser.add_argument("--appearance-object-steps", type=int, default=2400)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print("R440_GROWTH_PROBE_ACTIVE")
    if args.seed_grid is not None:
        print("R440_SEED_GRID_ACTIVE")
        grid = evaluate_r440_seed_grid(
            seeds=args.seed_grid,
            batch_size=args.batch_size,
            steps_per_stage=args.steps_per_stage,
        )
        payload = {
            "passed": grid.passed,
            "seed_grid": {
                "seeds": list(grid.seeds),
                "passed": grid.passed,
                "results": {str(seed): curriculum_payload(result) for seed, result in grid.results.items()},
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        return 0 if grid.passed else 1

    if args.sensory_bridge:
        print("R440_SENSORY_BRIDGE_ACTIVE")
        _bridge, _renderer, sensory_result = train_r440_sensory_bridge(
            seed=args.seed,
            batch_size=args.batch_size // 2,
            steps=args.sensory_steps,
        )
        payload = {
            "passed": sensory_result.passed,
            "sensory_bridge": {
                "seed": sensory_result.seed,
                "passed": sensory_result.passed,
                "metrics": sensory_result.metrics,
                "examples": sensory_result.examples,
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        return 0 if sensory_result.passed else 1

    if args.variable_object_files:
        print("R440_VARIABLE_OBJECT_FILES_ACTIVE")
        _bridge, _renderer, object_result = train_r440_variable_object_files(
            seed=args.seed,
            batch_size=args.batch_size // 2,
            steps=args.variable_object_steps,
        )
        payload = {
            "passed": object_result.passed,
            "variable_object_files": {
                "seed": object_result.seed,
                "passed": object_result.passed,
                "metrics": object_result.metrics,
                "examples": object_result.examples,
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        return 0 if object_result.passed else 1

    if args.appearance_object_files:
        print("R440_APPEARANCE_OBJECT_FILES_ACTIVE")
        _bridge, _renderer, appearance_result = train_r440_appearance_object_files(
            seed=args.seed,
            batch_size=args.batch_size // 2,
            steps=args.appearance_object_steps,
        )
        payload = {
            "passed": appearance_result.passed,
            "appearance_object_files": {
                "seed": appearance_result.seed,
                "passed": appearance_result.passed,
                "metrics": appearance_result.metrics,
                "examples": appearance_result.examples,
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        return 0 if appearance_result.passed else 1

    model, curriculum, result = train_r440_curriculum(
        seed=args.seed,
        batch_size=args.batch_size,
        steps_per_stage=args.steps_per_stage,
        stages=R440_CURRICULUM_V1,
    )
    verbalizer_payload = None
    passed = result.passed
    if args.verbalizer == "tiny":
        print("R440_TINY_VERBALIZER_ACTIVE")
        _verbalizer, verbalizer_result = train_r440_tiny_verbalizer(
            model,
            curriculum,
            seed=args.seed,
            steps=args.verbalizer_steps,
        )
        verbalizer_payload = {
            "seed": verbalizer_result.seed,
            "steps": verbalizer_result.steps,
            "passed": verbalizer_result.passed,
            "metrics": verbalizer_result.metrics,
            "examples": verbalizer_result.examples,
        }
        passed = passed and verbalizer_result.passed
    room_layout_payload = None
    if args.room_layout:
        print("R440_ROOM_LAYOUT_ACTIVE")
        room_layout_result = evaluate_r440_room_layout(model, seed=args.seed + 5000)
        room_layout_payload = {
            "seed": room_layout_result.seed,
            "passed": room_layout_result.passed,
            "metrics": room_layout_result.metrics,
            "examples": room_layout_result.examples,
        }
        passed = passed and room_layout_result.passed
    room_update_payload = None
    if args.room_update:
        print("R440_ROOM_UPDATE_ACTIVE")
        room_update_result = evaluate_r440_room_update(model, seed=args.seed + 6000)
        room_update_payload = {
            "seed": room_update_result.seed,
            "passed": room_update_result.passed,
            "metrics": room_update_result.metrics,
            "examples": room_update_result.examples,
        }
        passed = passed and room_update_result.passed
    room_relations_payload = None
    if args.room_relations:
        print("R440_ROOM_RELATIONS_ACTIVE")
        room_relations_result = evaluate_r440_room_relations(model, seed=args.seed + 7000)
        room_relations_payload = {
            "seed": room_relations_result.seed,
            "passed": room_relations_result.passed,
            "metrics": room_relations_result.metrics,
            "examples": room_relations_result.examples,
        }
        passed = passed and room_relations_result.passed
    room_sequential_payload = None
    if args.room_sequential_updates:
        print("R440_ROOM_SEQUENTIAL_UPDATES_ACTIVE")
        room_sequential_result = evaluate_r440_room_sequential_updates(model, seed=args.seed + 8000)
        room_sequential_payload = {
            "seed": room_sequential_result.seed,
            "passed": room_sequential_result.passed,
            "metrics": room_sequential_result.metrics,
            "examples": room_sequential_result.examples,
        }
        passed = passed and room_sequential_result.passed
    payload = curriculum_payload(result)
    payload["passed"] = passed
    payload["verbalizer"] = verbalizer_payload
    payload["room_layout"] = room_layout_payload
    payload["room_relations"] = room_relations_payload
    payload["room_sequential_updates"] = room_sequential_payload
    payload["room_update"] = room_update_payload
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
