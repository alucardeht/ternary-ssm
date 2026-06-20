#!/usr/bin/env python3
"""Analyze R405 DPTH A0/A1/A2 training logs.

The gate is intentionally conservative: a DPTH packet arm must improve
held-out CE per priced second against both the ordinary baseline and the
loss-proxy control before it can be promoted beyond a validation arm.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


TRAIN_RE = re.compile(
    r"^\[TRAIN\] step=(?P<step>\d+) loss=(?P<loss>[-+0-9.eE]+) "
    r"lr=(?P<lr>[-+0-9.eE]+) tok/s=(?P<toks>[-+0-9.eE]+)"
)
EVAL_RE = re.compile(
    r"^\[EVAL\] step=(?P<step>\d+) eval_loss=(?P<eval>[-+0-9.eE]+) "
    r"train_loss=(?P<train>[-+0-9.eE]+) gap=(?P<gap>[-+0-9.eE]+)"
)
DPTH_RE = re.compile(
    r"^\[DPTH_GATE\] step=(?P<step>\d+) mode=(?P<mode>[a-z-]+).*?"
    r"selected_score=(?P<score>[-+0-9.eE]+).*?"
    r"pool_score=(?P<pool_score>[-+0-9.eE]+).*?"
    r"loss_proxy=(?P<loss_proxy>[-+0-9.eE]+).*?"
    r"boundary_density=(?P<boundary>[-+0-9.eE]+).*?"
    r"selection_overhead=(?P<overhead>[-+0-9.eE]+)"
)


def parse_log(path: Path) -> dict:
    train = []
    evals = []
    dpth = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := TRAIN_RE.match(line):
            train.append({k: float(v) for k, v in m.groupdict().items()})
        elif m := EVAL_RE.match(line):
            evals.append({k: float(v) for k, v in m.groupdict().items()})
        elif m := DPTH_RE.match(line):
            row = m.groupdict()
            row["step"] = float(row["step"])
            for key in ("score", "pool_score", "loss_proxy", "boundary", "overhead"):
                row[key] = float(row[key])
            dpth.append(row)

    mode = "baseline"
    if dpth:
        mode = str(dpth[-1]["mode"])
    elif "loss-control" in path.name:
        mode = "loss-control"
    elif "packet" in path.name:
        mode = "packet"

    first_eval = evals[0] if evals else None
    last_eval = evals[-1] if evals else None
    mean_toks = sum(row["toks"] for row in train) / len(train) if train else math.nan
    mean_overhead = sum(row["overhead"] for row in dpth) / len(dpth) if dpth else 0.0
    progress = math.nan
    if first_eval and last_eval and last_eval["step"] > first_eval["step"]:
        progress = first_eval["eval"] - last_eval["eval"]

    priced_progress = math.nan
    if math.isfinite(progress) and math.isfinite(mean_toks) and mean_toks > 0.0:
        priced_progress = progress * mean_toks / (1.0 + max(mean_overhead, 0.0))

    return {
        "path": str(path),
        "mode": mode,
        "train_points": len(train),
        "eval_points": len(evals),
        "dpth_points": len(dpth),
        "first_eval": first_eval,
        "last_eval": last_eval,
        "eval_progress": progress,
        "mean_toks_per_sec": mean_toks,
        "mean_selection_overhead": mean_overhead,
        "priced_progress_proxy": priced_progress,
        "last_dpth": dpth[-1] if dpth else None,
    }


def ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1.0e-12:
        return math.nan
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--min-ratio", type=float, default=1.10)
    args = ap.parse_args()

    arms = [parse_log(path) for path in args.logs]
    by_mode = {arm["mode"]: arm for arm in arms}
    packet = by_mode.get("packet")
    baseline = by_mode.get("baseline")
    loss_control = by_mode.get("loss-control")

    packet_vs_baseline = math.nan
    packet_vs_loss = math.nan
    decision = "INSUFFICIENT"
    reason = "requires baseline, packet, loss-control, and at least two eval points per arm"

    if packet and baseline and loss_control:
        if min(arm["eval_points"] for arm in (packet, baseline, loss_control)) >= 2:
            priced = [arm["priced_progress_proxy"] for arm in (packet, baseline, loss_control)]
            progress = [arm["eval_progress"] for arm in (packet, baseline, loss_control)]
            if any((not math.isfinite(v) or v <= 0.0) for v in priced + progress):
                decision = "INSUFFICIENT"
                reason = "all arms need positive held-out CE progress before ratio pricing is meaningful"
            else:
                packet_vs_baseline = ratio(packet["priced_progress_proxy"], baseline["priced_progress_proxy"])
                packet_vs_loss = ratio(packet["priced_progress_proxy"], loss_control["priced_progress_proxy"])
                if packet_vs_baseline >= args.min_ratio and packet_vs_loss >= args.min_ratio:
                    decision = "DPTH_PACKET_PASSES_TRAINING_GATE"
                    reason = "packet arm clears priced progress ratio against both controls"
                else:
                    decision = "DPTH_PACKET_FAILS_TRAINING_GATE"
                    reason = "packet arm does not clear priced progress ratio against both controls"

    print(json.dumps({
        "decision": decision,
        "reason": reason,
        "min_ratio": args.min_ratio,
        "packet_vs_baseline": packet_vs_baseline,
        "packet_vs_loss_control": packet_vs_loss,
        "arms": arms,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
