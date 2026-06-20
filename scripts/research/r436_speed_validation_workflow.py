#!/usr/bin/env python3
"""Run and analyze an R436 speed-to-heldout-progress validation workflow.

This workflow exists because short single-seed throughput smokes are not enough
to decide training speed. It separates train and heldout data, runs repeated
ABBA arms, requires candidate activation, measures heldout CE progress per
wall-clock second, and blocks claims that sit inside the measured noise margin.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
TRAIN_RE = re.compile(
    r"\[TRAIN\] step=(?P<step>\d+) loss=(?P<loss>[0-9.eE+-]+).*?"
    r"tok/s=(?P<toks>[0-9.eE+-]+) vram=(?P<vram>[0-9.eE+-]+)MB"
)
EVAL_RE = re.compile(
    r"\[EVAL\] step=(?P<step>\d+) eval_loss=(?P<eval>[0-9.eE+-]+) "
    r"train_loss=(?P<train>[0-9.eE+-]+) gap=(?P<gap>[0-9.eE+-]+)"
)
HELDOUT_RE = re.compile(
    r"\[HELDOUT_EVAL\] tag=(?P<tag>\S+) step=(?P<step>\d+) "
    r"batches=(?P<batches>\d+) dataset=(?P<dataset>.+?) "
    r"loss=(?P<loss>[0-9.eE+-]+)"
)


@dataclass
class ArmSpec:
    name: str
    kind: str
    seed: int
    repeat: int
    extra_flags: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=43650)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--heldout-evals", type=int, default=3)
    parser.add_argument("--train-tokens", type=int, default=524288)
    parser.add_argument("--heldout-tokens", type=int, default=131072)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-speed-gain", type=float, default=0.05)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    parser.add_argument("--min-heldout-progress", type=float, default=0.01)
    parser.add_argument("--min-progress-rate-gain", type=float, default=0.05)
    parser.add_argument("--noise-floor", type=float, default=0.03)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_dataset_tokens(path: Path) -> list[int]:
    raw = path.read_bytes()
    offset = 0
    if len(raw) >= DATASET_HEADER_BYTES:
        magic = int.from_bytes(raw[0:4], "little")
        if magic == DATASET_MAGIC:
            offset = DATASET_HEADER_BYTES
    data = raw[offset:]
    if len(data) % 2 != 0:
        data = data[:-1]
    return [
        int.from_bytes(data[i : i + 2], "little")
        for i in range(0, len(data), 2)
    ]


def write_raw_tokens(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bytearray()
    for token in tokens:
        out.extend(int(token).to_bytes(2, "little", signed=False))
    path.write_bytes(bytes(out))


def prepare_split(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    tokens = read_dataset_tokens(args.source_dataset)
    required = args.train_tokens + args.heldout_tokens
    if len(tokens) < required:
        raise ValueError(
            f"source dataset has {len(tokens)} tokens, need at least {required}"
        )
    train_tokens = tokens[: args.train_tokens]
    heldout_tokens = tokens[-args.heldout_tokens :]
    data_dir = args.output_dir / "data"
    train_path = data_dir / "train_prefix.bin"
    heldout_path = data_dir / "heldout_suffix.bin"
    write_raw_tokens(train_path, train_tokens)
    write_raw_tokens(heldout_path, heldout_tokens)
    return train_path, heldout_path, {
        "source_dataset": str(args.source_dataset),
        "source_tokens": len(tokens),
        "train_tokens": len(train_tokens),
        "heldout_tokens": len(heldout_tokens),
        "train_path": str(train_path),
        "heldout_path": str(heldout_path),
        "split_policy": "train_prefix_heldout_suffix_no_overlap",
    }


def parse_train_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in TRAIN_RE.finditer(text):
        rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "tok_s": float(match.group("toks")),
                "vram_mb": float(match.group("vram")),
            }
        )
    return rows


def latest_eval_loss(text: str) -> float | None:
    matches = list(EVAL_RE.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group("eval"))


def latest_heldout_loss(text: str) -> float | None:
    matches = list(HELDOUT_RE.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group("loss"))


def finite_mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def finite_stdev(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return 0.0
    return float(statistics.stdev(vals))


def command_string(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(command_string(command) + "\n", encoding="utf-8")
        return 0, 0.0
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start
    return int(proc.returncode), elapsed


def build_train_command(
    train_exe: Path,
    dataset: Path,
    steps: int,
    checkpoint: Path | None,
    batch: int,
    seed: int,
    log_every: int,
    extra_flags: list[str] | None = None,
    lr_base: float | None = None,
    checkpoint_load_only: bool = False,
    heldout_dataset: Path | None = None,
    heldout_eval_batches: int = 0,
    heldout_seed: int | None = None,
) -> list[str]:
    command = [str(train_exe), str(dataset), str(steps)]
    if checkpoint is not None:
        command.append(str(checkpoint))
    command += [
        "--batch",
        str(batch),
        "--seed",
        str(seed),
        "--teaching-off",
        "--log-every",
        str(log_every),
        "--eval-every",
        "0",
        "--eval-batches",
        "1",
    ]
    if lr_base is not None:
        command += ["--lr-base", f"{lr_base:.9g}"]
    if checkpoint_load_only:
        command.append("--checkpoint-load-only")
    if heldout_dataset is not None and heldout_eval_batches > 0:
        command += [
            "--heldout-dataset",
            str(heldout_dataset),
            "--heldout-eval-batches",
            str(heldout_eval_batches),
        ]
        if heldout_seed is not None:
            command += ["--heldout-eval-seed", str(heldout_seed)]
    if extra_flags:
        command += extra_flags
    return command


def run_standalone_heldout_eval(
    train_exe: Path,
    model_dir: Path,
    heldout_path: Path,
    batch: int,
    eval_seed: int,
    eval_batches: int,
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    command = build_train_command(
        train_exe=train_exe,
        dataset=heldout_path,
        steps=0,
        checkpoint=None,
        batch=batch,
        seed=eval_seed,
        log_every=1,
        heldout_dataset=heldout_path,
        heldout_eval_batches=eval_batches,
        heldout_seed=eval_seed,
    )
    exit_code, elapsed = run_command(command, model_dir, log_path, dry_run)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    loss = latest_heldout_loss(text)
    return {
        "command": command_string(command),
        "log": str(log_path),
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "heldout_loss": loss,
    }


def summarize_arm(
    spec: ArmSpec,
    text: str,
    elapsed: float,
    exit_code: int,
    log_path: Path,
    checkpoint_path: Path | None,
    heldout_evals: list[dict[str, Any]],
    initial_heldout_mean: float | None,
) -> dict[str, Any]:
    rows = parse_train_rows(text)
    final_train_loss = rows[-1]["loss"] if rows else None
    final_tok_s = rows[-1]["tok_s"] if rows else None
    final_vram = rows[-1]["vram_mb"] if rows else None
    heldout_values = [
        float(item["heldout_loss"])
        for item in heldout_evals
        if isinstance(item.get("heldout_loss"), (int, float))
    ]
    heldout_mean = finite_mean(heldout_values)
    heldout_progress = None
    progress_per_second = None
    if heldout_mean is not None and initial_heldout_mean is not None:
        heldout_progress = initial_heldout_mean - heldout_mean
        progress_per_second = heldout_progress / elapsed if elapsed > 0.0 else None
    return {
        "name": spec.name,
        "kind": spec.kind,
        "seed": spec.seed,
        "repeat": spec.repeat,
        "log": str(log_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "final_train_loss": final_train_loss,
        "final_tok_s": final_tok_s,
        "final_vram_mb": final_vram,
        "heldout_eval_loss_mean": heldout_mean,
        "heldout_eval_loss_values": heldout_values,
        "initial_heldout_loss_mean": initial_heldout_mean,
        "heldout_progress": heldout_progress,
        "heldout_progress_per_second": progress_per_second,
        "martingale_active": "[AMORTIZED_CE_MARTINGALE_TRAINING] active=1" in text,
        "martingale_applied": bool(
            re.search(r"\[AMORTIZED_CE_MARTINGALE_TRAINING\].*applied_layers=[1-9]", text)
        ),
        "variance_pass": bool(
            re.search(r"\[VARIANCE_PRICED_CREDIT_GATE\].*identifying=1 pass=1", text)
        ),
        "heldout_evals": heldout_evals,
    }


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.repo_root / "src" / "model_v12"
    train_exe = model_dir / "train.exe"
    if not train_exe.exists() and not args.dry_run:
        raise FileNotFoundError(f"missing train executable: {train_exe}")

    train_path, heldout_path, split = prepare_split(args)
    candidate_flags = [
        "--variance-priced-credit-gate",
        "--variance-priced-sample-fraction",
        "0.250",
        "--variance-priced-score-fraction",
        "0.050",
        "--variance-priced-verify-fraction",
        "0.050",
        "--variance-priced-overhead",
        "0.020",
        "--amortized-ce-martingale-training",
        "--amortized-ce-martingale-min-descent-per-paid-work",
        "1.111",
    ]

    manifest: dict[str, Any] = {
        "schema": "r436_speed_validation_workflow.v2",
        "round": "R436",
        "claim_scope": "speed_to_heldout_progress_validation_not_official_7b_projection",
        "evaluation_policy": "checkpointless_inline_heldout_eval_before_final_save",
        "split": split,
        "steps": args.steps,
        "batch": args.batch,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "repeats": args.repeats,
        "heldout_evals": args.heldout_evals,
        "warmup_steps": args.warmup_steps,
        "thresholds": {
            "min_speed_gain": args.min_speed_gain,
            "quality_tolerance": args.quality_tolerance,
            "min_heldout_progress": args.min_heldout_progress,
            "min_progress_rate_gain": args.min_progress_rate_gain,
            "noise_floor": args.noise_floor,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.warmup_steps > 0:
        for kind, flags in (("baseline", []), ("candidate", candidate_flags)):
            warmup_log = args.output_dir / "warmup" / f"{kind}.log"
            command = build_train_command(
                train_exe,
                train_path,
                args.warmup_steps,
                None,
                args.batch,
                args.seed_start,
                1,
                flags,
            )
            run_command(command, model_dir, warmup_log, args.dry_run)

    initial_evals = [
        run_standalone_heldout_eval(
            train_exe=train_exe,
            model_dir=model_dir,
            heldout_path=heldout_path,
            batch=args.batch,
            eval_seed=args.seed_start + 10000,
            eval_batches=args.heldout_evals,
            log_path=args.output_dir / "initial_heldout" / "eval_0.log",
            dry_run=args.dry_run,
        )
    ]
    initial_heldout_mean = finite_mean(
        [
            float(item["heldout_loss"])
            for item in initial_evals
            if isinstance(item.get("heldout_loss"), (int, float))
        ]
    )

    arms: list[dict[str, Any]] = []
    order_template = [("baseline", []), ("candidate", candidate_flags), ("candidate", candidate_flags), ("baseline", [])]
    for seed_index in range(args.seed_count):
        seed = args.seed_start + seed_index
        for repeat in range(args.repeats):
            for order_index, (kind, flags) in enumerate(order_template):
                name = f"{kind}_s{seed}_r{repeat}_o{order_index}"
                spec = ArmSpec(name=name, kind=kind, seed=seed, repeat=repeat, extra_flags=flags)
                arm_dir = args.output_dir / "arms" / name
                train_log = arm_dir / "train.log"
                command = build_train_command(
                    train_exe=train_exe,
                    dataset=train_path,
                    steps=args.steps,
                    checkpoint=None,
                    batch=args.batch,
                    seed=seed,
                    log_every=args.log_every,
                    extra_flags=flags,
                    heldout_dataset=heldout_path,
                    heldout_eval_batches=args.heldout_evals,
                    heldout_seed=args.seed_start + 20000 + seed_index * 100 + repeat,
                )
                exit_code, elapsed = run_command(command, model_dir, train_log, args.dry_run)
                text = train_log.read_text(encoding="utf-8", errors="replace") if train_log.exists() else ""
                heldout_loss = latest_heldout_loss(text)
                heldout_evals = [
                    {
                        "command": "inline_final_heldout_eval",
                        "log": str(train_log),
                        "exit_code": exit_code,
                        "elapsed_seconds": 0.0,
                        "heldout_loss": heldout_loss,
                    }
                ]
                arm = summarize_arm(
                    spec,
                    text,
                    elapsed,
                    exit_code,
                    train_log,
                    None,
                    heldout_evals,
                    initial_heldout_mean,
                )
                arm["command"] = command_string(command)
                arms.append(arm)

    result = analyze_result(args, manifest, arms, initial_evals)
    result_path = args.output_dir / "r436_speed_validation_analysis.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def mean_metric(arms: list[dict[str, Any]], key: str) -> float | None:
    vals = [
        float(arm[key])
        for arm in arms
        if isinstance(arm.get(key), (int, float)) and math.isfinite(float(arm[key]))
    ]
    return finite_mean(vals)


def analyze_result(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    arms: list[dict[str, Any]],
    initial_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = [arm for arm in arms if arm["kind"] == "baseline"]
    candidate = [arm for arm in arms if arm["kind"] == "candidate"]
    baseline_elapsed = mean_metric(baseline, "elapsed_seconds")
    candidate_elapsed = mean_metric(candidate, "elapsed_seconds")
    baseline_progress_rate = mean_metric(baseline, "heldout_progress_per_second")
    candidate_progress_rate = mean_metric(candidate, "heldout_progress_per_second")
    baseline_heldout = mean_metric(baseline, "heldout_eval_loss_mean")
    candidate_heldout = mean_metric(candidate, "heldout_eval_loss_mean")
    baseline_progress = mean_metric(baseline, "heldout_progress")
    candidate_progress = mean_metric(candidate, "heldout_progress")

    elapsed_ratio = (
        candidate_elapsed / baseline_elapsed
        if baseline_elapsed and candidate_elapsed
        else math.nan
    )
    progress_rate_ratio = (
        candidate_progress_rate / baseline_progress_rate
        if baseline_progress_rate and candidate_progress_rate and baseline_progress_rate > 0.0
        else math.nan
    )

    baseline_elapsed_values = [float(arm["elapsed_seconds"]) for arm in baseline]
    candidate_elapsed_values = [float(arm["elapsed_seconds"]) for arm in candidate]
    baseline_rel_noise = (
        finite_stdev(baseline_elapsed_values) / baseline_elapsed
        if baseline_elapsed
        else 0.0
    )
    candidate_rel_noise = (
        finite_stdev(candidate_elapsed_values) / candidate_elapsed
        if candidate_elapsed
        else 0.0
    )
    measured_noise = max(args.noise_floor, 2.0 * baseline_rel_noise, 2.0 * candidate_rel_noise)

    candidate_activation_ok = bool(candidate) and all(
        arm["martingale_active"] and arm["martingale_applied"] and arm["variance_pass"]
        for arm in candidate
    )
    all_exited = all(arm["exit_code"] == 0 for arm in arms)
    enough_repeats = len(baseline) >= 2 and len(candidate) >= 2
    heldout_identifying = (
        baseline_progress is not None
        and candidate_progress is not None
        and max(baseline_progress, candidate_progress) >= args.min_heldout_progress
    )
    quality_nonregression = (
        baseline_heldout is not None
        and candidate_heldout is not None
        and candidate_heldout <= baseline_heldout + args.quality_tolerance
    )
    speed_gain = 1.0 - elapsed_ratio if math.isfinite(elapsed_ratio) else math.nan
    progress_rate_gain = (
        progress_rate_ratio - 1.0 if math.isfinite(progress_rate_ratio) else math.nan
    )

    if not all_exited:
        decision = "FAIL_R436_SPEED_VALIDATION_RUNTIME_FAILURE"
    elif not candidate_activation_ok:
        decision = "FAIL_R436_SPEED_VALIDATION_CANDIDATE_NOT_ACTIVE"
    elif not enough_repeats:
        decision = "NO_DECISION_R436_SPEED_VALIDATION_INSUFFICIENT_REPEATS"
    elif not heldout_identifying:
        decision = "NO_DECISION_R436_SPEED_VALIDATION_HELDOUT_NON_IDENTIFYING"
    elif not quality_nonregression:
        decision = "FAIL_R436_SPEED_VALIDATION_HELDOUT_REGRESSION"
    elif speed_gain >= max(args.min_speed_gain, measured_noise) and progress_rate_gain >= args.min_progress_rate_gain:
        decision = "PASS_R436_SPEED_TO_HELDOUT_PROGRESS"
    elif abs(speed_gain) < measured_noise:
        decision = "NO_DECISION_R436_SPEED_VALIDATION_WITHIN_NOISE_MARGIN"
    else:
        decision = "FAIL_R436_SPEED_VALIDATION_SPEED_NEGATIVE"

    return {
        "decision": decision,
        "schema": manifest["schema"],
        "claim_scope": manifest["claim_scope"],
        "manifest": manifest,
        "initial_heldout_evals": initial_evals,
        "arms": arms,
        "summary": {
            "baseline_arm_count": len(baseline),
            "candidate_arm_count": len(candidate),
            "baseline_elapsed_mean": baseline_elapsed,
            "candidate_elapsed_mean": candidate_elapsed,
            "elapsed_ratio_candidate_over_baseline": elapsed_ratio,
            "speed_gain_fraction": speed_gain,
            "baseline_heldout_loss_mean": baseline_heldout,
            "candidate_heldout_loss_mean": candidate_heldout,
            "baseline_heldout_progress_mean": baseline_progress,
            "candidate_heldout_progress_mean": candidate_progress,
            "baseline_progress_per_second": baseline_progress_rate,
            "candidate_progress_per_second": candidate_progress_rate,
            "progress_rate_ratio_candidate_over_baseline": progress_rate_ratio,
            "progress_rate_gain_fraction": progress_rate_gain,
            "measured_noise_margin_fraction": measured_noise,
            "baseline_relative_noise": baseline_rel_noise,
            "candidate_relative_noise": candidate_rel_noise,
            "candidate_activation_ok": candidate_activation_ok,
            "heldout_identifying": heldout_identifying,
            "quality_nonregression": quality_nonregression,
            "all_exited": all_exited,
            "enough_repeats": enough_repeats,
        },
        "interpretation": [
            "A speed claim requires heldout CE progress, not train-loss movement alone.",
            "Candidate activation is mandatory but insufficient.",
            "Effects inside the measured noise margin are classified as no decision.",
            "This workflow does not update official 7B projections by itself.",
        ],
    }


def main() -> int:
    args = parse_args()
    result = run_workflow(args)
    print(
        "[R436_SPEED_VALIDATION] "
        f"decision={result['decision']} "
        f"elapsed_ratio={result['summary']['elapsed_ratio_candidate_over_baseline']} "
        f"progress_rate_ratio={result['summary']['progress_rate_ratio_candidate_over_baseline']} "
        f"output={args.output_dir / 'r436_speed_validation_analysis.json'}"
    )
    if args.dry_run:
        return 0
    if result["decision"].startswith("FAIL_"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
