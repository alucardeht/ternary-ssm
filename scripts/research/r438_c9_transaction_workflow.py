#!/usr/bin/env python3
"""Run and analyze R438 C9 CE transaction block validation.

This workflow mirrors the repaired R436/R438 ABBA heldout-progress harness, but
the candidate is observer-only: it enables CE transaction block receipts and
checks whether the observed accept/reject/rollback economics are identifying.
It does not promote a training-speed claim by itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def load_r436_module() -> Any:
    path = Path(__file__).with_name("r436_speed_validation_workflow.py")
    spec = importlib.util.spec_from_file_location("r436_speed_validation_workflow", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared workflow helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R436 = load_r436_module()

CE_TX_SUMMARY_RE = re.compile(
    r"\[CE_TX_OBSERVER\] summary "
    r"k=(?P<k>\d+) "
    r"observed=(?P<observed>\d+) "
    r"accepted=(?P<accepted>\d+) "
    r"rejected=(?P<rejected>\d+) "
    r"rollback=(?P<rollback>\d+) "
    r"accept_density=(?P<accept_density>[0-9.eE+-]+) "
    r"rollback_fraction=(?P<rollback_fraction>[0-9.eE+-]+) "
    r"reject_fraction=(?P<reject_fraction>[0-9.eE+-]+) "
    r"verifier_fraction=(?P<verifier_fraction>[0-9.eE+-]+) "
    r"priced_ratio=(?P<priced_ratio>[0-9.eE+-]+) "
    r"last_margin=(?P<last_margin>[0-9.eE+-]+) "
    r"pass=(?P<pass>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=43860)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--heldout-evals", type=int, default=1)
    parser.add_argument("--train-tokens", type=int, default=65536)
    parser.add_argument("--heldout-tokens", type=int, default=32768)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--max-rollback-fraction", type=float, default=0.10)
    parser.add_argument("--verifier-fraction", type=float, default=0.05)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    parser.add_argument("--min-heldout-progress", type=float, default=0.01)
    parser.add_argument("--min-progress-rate-gain", type=float, default=0.05)
    parser.add_argument("--noise-floor", type=float, default=0.03)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def candidate_flags(args: argparse.Namespace) -> list[str]:
    return [
        "--r438-ce-transaction-blocks",
        "--r438-ce-tx-k",
        str(args.k),
        "--r438-ce-tx-min-margin",
        f"{args.min_margin:.6f}",
        "--r438-ce-tx-max-rollback-fraction",
        f"{args.max_rollback_fraction:.6f}",
        "--r438-ce-tx-verifier-fraction",
        f"{args.verifier_fraction:.6f}",
    ]


def parse_ce_tx_summary(text: str) -> dict[str, Any] | None:
    matches = list(CE_TX_SUMMARY_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    out: dict[str, Any] = {}
    for key, value in match.groupdict().items():
        if key in {"k", "observed", "accepted", "rejected", "rollback", "pass"}:
            out[key] = int(value)
        else:
            out[key] = float(value)
    out["passed"] = out.pop("pass") == 1
    return out


def summarize_c9_arm(
    spec: Any,
    text: str,
    elapsed: float,
    exit_code: int,
    log_path: Path,
    heldout_evals: list[dict[str, Any]],
    initial_heldout_mean: float | None,
) -> dict[str, Any]:
    arm = R436.summarize_arm(
        spec,
        text,
        elapsed,
        exit_code,
        log_path,
        None,
        heldout_evals,
        initial_heldout_mean,
    )
    summary = parse_ce_tx_summary(text)
    arm["ce_tx_active"] = "[CE_TX_OBSERVER] active=1" in text
    arm["ce_tx_summary"] = summary
    arm["ce_tx_observed"] = bool(summary and summary.get("observed", 0) > 0)
    arm["ce_tx_pass"] = bool(summary and summary.get("passed"))
    return arm


def mean_metric(arms: list[dict[str, Any]], key: str) -> float | None:
    return R436.mean_metric(arms, key)


def stdev_metric(arms: list[dict[str, Any]], key: str) -> float:
    vals = [
        float(arm[key])
        for arm in arms
        if isinstance(arm.get(key), (int, float)) and math.isfinite(float(arm[key]))
    ]
    return R436.finite_stdev(vals)


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
        if baseline_elapsed and candidate_elapsed and baseline_elapsed > 0.0
        else None
    )
    speed_gain = 1.0 - elapsed_ratio if elapsed_ratio is not None else 0.0
    progress_rate_ratio = (
        candidate_progress_rate / baseline_progress_rate
        if baseline_progress_rate and candidate_progress_rate and baseline_progress_rate > 0.0
        else None
    )
    progress_rate_gain = (
        progress_rate_ratio - 1.0 if progress_rate_ratio is not None else 0.0
    )
    baseline_rate_noise = stdev_metric(baseline, "heldout_progress_per_second")
    candidate_rate_noise = stdev_metric(candidate, "heldout_progress_per_second")
    baseline_rel_noise = (
        baseline_rate_noise / abs(baseline_progress_rate)
        if baseline_progress_rate and baseline_progress_rate != 0.0
        else 0.0
    )
    candidate_rel_noise = (
        candidate_rate_noise / abs(candidate_progress_rate)
        if candidate_progress_rate and candidate_progress_rate != 0.0
        else 0.0
    )
    measured_noise = max(args.noise_floor, 2.0 * baseline_rel_noise, 2.0 * candidate_rel_noise)
    candidate_observer_ok = bool(candidate) and all(
        arm["ce_tx_active"] and arm["ce_tx_observed"] for arm in candidate
    )
    candidate_economics_ok = bool(candidate) and all(arm["ce_tx_pass"] for arm in candidate)
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

    if not all_exited:
        decision = "FAIL_R438_C9_TRANSACTION_RUNTIME_FAILURE"
    elif not candidate_observer_ok:
        decision = "FAIL_R438_C9_TRANSACTION_OBSERVER_NOT_ACTIVE"
    elif not candidate_economics_ok:
        decision = "FAIL_R438_C9_TRANSACTION_OBSERVER_ECONOMICS"
    elif not enough_repeats:
        decision = "NO_DECISION_R438_C9_TRANSACTION_INSUFFICIENT_REPEATS"
    elif not heldout_identifying:
        decision = "NO_DECISION_R438_C9_TRANSACTION_HELDOUT_NON_IDENTIFYING"
    elif not quality_nonregression:
        decision = "FAIL_R438_C9_TRANSACTION_HELDOUT_REGRESSION"
    else:
        decision = "PASS_R438_C9_TRANSACTION_OBSERVER_GATE"

    if progress_rate_gain >= max(args.min_progress_rate_gain, measured_noise):
        speed_decision = "PASS_R438_C9_PROGRESS_RATE_SIGNAL"
    elif abs(progress_rate_gain) < measured_noise:
        speed_decision = "NO_DECISION_R438_C9_PROGRESS_RATE_WITHIN_NOISE"
    else:
        speed_decision = "FAIL_R438_C9_PROGRESS_RATE_NEGATIVE"

    return {
        "decision": decision,
        "speed_decision": speed_decision,
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
            "baseline_progress_rate_mean": baseline_progress_rate,
            "candidate_progress_rate_mean": candidate_progress_rate,
            "progress_rate_ratio_candidate_over_baseline": progress_rate_ratio,
            "progress_rate_gain_fraction": progress_rate_gain,
            "measured_noise_margin_fraction": measured_noise,
            "baseline_relative_noise": baseline_rel_noise,
            "candidate_relative_noise": candidate_rel_noise,
            "candidate_observer_ok": candidate_observer_ok,
            "candidate_economics_ok": candidate_economics_ok,
            "heldout_identifying": heldout_identifying,
            "quality_nonregression": quality_nonregression,
            "all_exited": all_exited,
            "enough_repeats": enough_repeats,
        },
        "interpretation": [
            "C9 speed promotion requires heldout CE progress per wall-clock, not transaction logs alone.",
            "This workflow treats CE_TX_* as observer economics because the current path is observer-only.",
            "Effects inside the measured noise margin are no-decision results.",
            "This workflow does not update official 7B projections by itself.",
        ],
    }


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.repo_root / "src" / "model_v12"
    train_exe = model_dir / "train.exe"
    if not train_exe.exists():
        raise FileNotFoundError(f"missing train executable: {train_exe}")

    train_path, heldout_path, split = R436.prepare_split(args)
    flags = candidate_flags(args)
    manifest: dict[str, Any] = {
        "schema": "r438_c9_transaction_workflow.v1",
        "round": "R438",
        "finding": "C9 CE transaction blocks",
        "claim_scope": "observer_gate_and_heldout_progress_context_not_official_7b_projection",
        "evaluation_policy": "checkpointless_inline_heldout_eval_before_final_save",
        "split": split,
        "candidate_flags": flags,
        "arm_order": "ABBA_per_seed",
        "activation_required": "CE_TX_OBSERVER active=1 summary observed>0",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.warmup_steps > 0:
        for kind, extra in (("baseline", []), ("candidate", flags)):
            command = R436.build_train_command(
                train_exe,
                train_path,
                args.warmup_steps,
                None,
                args.batch,
                args.seed_start,
                args.log_every,
                extra,
            )
            R436.run_command(command, model_dir, args.output_dir / "warmup" / f"{kind}.log", args.dry_run)

    initial_evals = [
        R436.run_standalone_heldout_eval(
            train_exe,
            model_dir,
            heldout_path,
            args.batch,
            args.seed_start + 1000 + i,
            args.heldout_evals,
            args.output_dir / "initial_heldout" / f"eval_{i}.log",
            args.dry_run,
        )
        for i in range(max(1, args.seed_count))
    ]
    initial_mean = R436.finite_mean(
        [float(row["heldout_loss"]) for row in initial_evals if row.get("heldout_loss") is not None]
    )

    arms: list[dict[str, Any]] = []
    order_template = [("baseline", []), ("candidate", flags), ("candidate", flags), ("baseline", [])]
    for seed_index in range(args.seed_count):
        seed = args.seed_start + seed_index
        for repeat in range(args.repeats):
            for order_index, (kind, extra) in enumerate(order_template):
                name = f"{kind}_s{seed}_r{repeat}_o{order_index}"
                spec = R436.ArmSpec(name=name, kind=kind, seed=seed, repeat=repeat, extra_flags=extra)
                arm_dir = args.output_dir / "arms" / name
                train_log = arm_dir / "train.log"
                command = R436.build_train_command(
                    train_exe=train_exe,
                    dataset=train_path,
                    steps=args.steps,
                    checkpoint=None,
                    batch=args.batch,
                    seed=seed,
                    log_every=args.log_every,
                    extra_flags=extra,
                    heldout_dataset=heldout_path,
                    heldout_eval_batches=args.heldout_evals,
                    heldout_seed=seed + 2000 + order_index,
                )
                exit_code, elapsed = R436.run_command(command, model_dir, train_log, args.dry_run)
                text = train_log.read_text(encoding="utf-8", errors="replace") if train_log.exists() else ""
                heldout_loss = R436.latest_heldout_loss(text)
                heldout_evals = [
                    {
                        "tag": "inline_final",
                        "heldout_loss": heldout_loss,
                        "log_path": str(train_log),
                    }
                ]
                arm = summarize_c9_arm(
                    spec,
                    text,
                    elapsed,
                    exit_code,
                    train_log,
                    heldout_evals,
                    initial_mean,
                )
                arm["command"] = R436.command_string(command)
                arms.append(arm)

    result = analyze_result(args, manifest, arms, initial_evals)
    result_path = args.output_dir / "r438_c9_transaction_analysis.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    result = run_workflow(args)
    output_path = args.output_dir / "r438_c9_transaction_analysis.json"
    print(
        "[R438_C9_TRANSACTION_WORKFLOW] "
        f"decision={result['decision']} "
        f"speed_decision={result['speed_decision']} "
        f"progress_rate_ratio={result['summary']['progress_rate_ratio_candidate_over_baseline']} "
        f"output={output_path}"
    )
    if args.dry_run:
        return 0
    if result["decision"].startswith("FAIL_"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
