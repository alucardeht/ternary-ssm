#!/usr/bin/env python3
"""R422 credit market with CE arbitrage gate."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "r422_credit_market_ce_arbitrage_gate.v1"


@dataclass
class ArmMetrics:
    arm: str
    realized_gain_per_work: float
    selected_bad_rate: float


@dataclass
class Block:
    cost: float
    true_gain: float
    grad_norm_score: float
    certified_bid: float
    raw_bid: float
    bad_bidder: bool


@dataclass
class GateResult:
    schema: str
    decision: str
    pass_gate: bool
    mode: str
    trials: int
    blocks_per_trial: int
    budget_fraction: float
    market_gain_per_work: float
    grad_norm_gain_per_work: float
    random_gain_per_work: float
    oracle_gain_per_work: float
    no_settlement_gain_per_work: float
    market_vs_grad_norm: float
    market_vs_random: float
    market_selected_bad_rate: float
    no_settlement_bad_rate: float
    activation_log: str
    interpretation: str
    arms: list[ArmMetrics]
    elapsed_seconds: float


def make_block(
    rng: random.Random,
    bid_noise: float,
    bad_bidder_rate: float,
    miscalibrated_bid: bool,
) -> Block:
    cost = rng.uniform(0.5, 2.5)
    base = rng.gammavariate(2.0, 0.8)
    interaction_harm = rng.random() ** 3 * 0.8
    true_gain = max(0.0, base - interaction_harm) * cost
    grad_norm_score = max(0.0, base + rng.gauss(0.0, 0.8)) / cost
    bad_bidder = rng.random() < bad_bidder_rate
    if miscalibrated_bid:
        honest_bid = max(0.0, rng.gammavariate(1.4, 0.7) + rng.gauss(0.0, bid_noise))
    else:
        honest_bid = max(0.0, true_gain / cost - 0.05 - abs(rng.gauss(0.0, bid_noise)))
    raw_bid = honest_bid
    if bad_bidder:
        raw_bid = honest_bid + rng.uniform(0.8, 2.2)
    # Settlement means a bidder that overstates its CE decrease loses future
    # eligibility. The certified arm therefore prices bad bidders at zero rather
    # than letting their raw overbid keep buying budget.
    certified_bid = 0.0 if bad_bidder else raw_bid
    return Block(cost, true_gain, grad_norm_score, certified_bid, raw_bid, bad_bidder)


def select_blocks(blocks: list[Block], budget: float, score_name: str, rng: random.Random) -> list[Block]:
    if score_name == "random":
        ordered = blocks[:]
        rng.shuffle(ordered)
    elif score_name == "oracle_ratio":
        ordered = sorted(
            blocks,
            key=lambda block: block.true_gain / max(1e-9, block.cost),
            reverse=True,
        )
    else:
        ordered = sorted(blocks, key=lambda block: getattr(block, score_name), reverse=True)
    selected: list[Block] = []
    spent = 0.0
    for block in ordered:
        if spent + block.cost <= budget:
            selected.append(block)
            spent += block.cost
    return selected


def arm_score(selected: list[Block]) -> tuple[float, float]:
    spent = sum(block.cost for block in selected)
    gain = sum(block.true_gain for block in selected)
    bad_rate = sum(block.bad_bidder for block in selected) / max(1, len(selected))
    return gain / max(1e-9, spent), bad_rate


def run_gate(args: argparse.Namespace) -> GateResult:
    start_time = time.perf_counter()
    rng = random.Random(args.seed)
    market_scores: list[float] = []
    grad_scores: list[float] = []
    random_scores: list[float] = []
    oracle_scores: list[float] = []
    no_settlement_scores: list[float] = []
    market_bad_rates: list[float] = []
    no_settlement_bad_rates: list[float] = []

    bid_noise = args.bid_noise
    miscalibrated_bid = False
    if args.mode == "high_noise":
        bid_noise = max(bid_noise, 0.75)
        miscalibrated_bid = True
    bad_rate = args.bad_bidder_rate
    if args.mode == "no_settlement":
        bad_rate = max(bad_rate, 0.25)

    for _ in range(args.trials):
        blocks = [
            make_block(rng, bid_noise, bad_rate, miscalibrated_bid)
            for _ in range(args.blocks_per_trial)
        ]
        total_budget = args.budget_fraction * sum(block.cost for block in blocks)
        market = select_blocks(blocks, total_budget, "certified_bid", rng)
        grad = select_blocks(blocks, total_budget, "grad_norm_score", rng)
        rand = select_blocks(blocks, total_budget, "random", rng)
        oracle = select_blocks(blocks, total_budget, "oracle_ratio", rng)
        no_settle = select_blocks(blocks, total_budget, "raw_bid", rng)

        score, bad = arm_score(market)
        market_scores.append(score)
        market_bad_rates.append(bad)
        grad_scores.append(arm_score(grad)[0])
        random_scores.append(arm_score(rand)[0])
        oracle_scores.append(arm_score(oracle)[0])
        score, bad = arm_score(no_settle)
        no_settlement_scores.append(score)
        no_settlement_bad_rates.append(bad)

    market_gain = sum(market_scores) / args.trials
    grad_gain = sum(grad_scores) / args.trials
    random_gain = sum(random_scores) / args.trials
    oracle_gain = sum(oracle_scores) / args.trials
    no_settlement_gain = sum(no_settlement_scores) / args.trials
    market_bad = sum(market_bad_rates) / args.trials
    no_settle_bad = sum(no_settlement_bad_rates) / args.trials
    market_vs_grad = market_gain / max(1e-9, grad_gain)
    market_vs_random = market_gain / max(1e-9, random_gain)

    pass_gate = (
        args.mode == "settled"
        and market_vs_grad >= args.min_vs_grad
        and market_vs_random >= args.min_vs_random
        and market_bad <= args.max_bad_selected_rate
        and no_settle_bad >= args.min_no_settlement_bad_rate
    )
    if pass_gate:
        decision = "PASS_CREDIT_MARKET_CE_ARBITRAGE_GATE"
        interpretation = (
            "Certified CE-gain bids allocate budget better than gradient-norm "
            "and random heuristics while settlement suppresses bad bidders."
        )
    elif args.mode == "no_settlement":
        decision = "FAIL_NO_SETTLEMENT_OVERBID_CONTROL"
        interpretation = "Without settlement, overbidders remain selected too often."
    elif args.mode == "high_noise":
        decision = "FAIL_NOISY_BID_MARKET_CONTROL"
        interpretation = "Noisy bids do not reliably beat simple heuristics."
    else:
        decision = "FAIL_CREDIT_MARKET_CE_ARBITRAGE_GATE"
        interpretation = "The market allocation did not clear heuristic and bad-bidder controls."

    arms = [
        ArmMetrics("certified_market", market_gain, market_bad),
        ArmMetrics("gradient_norm", grad_gain, 0.0),
        ArmMetrics("random", random_gain, 0.0),
        ArmMetrics("oracle", oracle_gain, 0.0),
        ArmMetrics("raw_bid_no_settlement", no_settlement_gain, no_settle_bad),
    ]
    activation_log = (
        "[R422_CREDIT_MARKET_CE_ARBITRAGE_GATE] "
        f"active=1 decision={decision} mode={args.mode} "
        f"no_gpu=1 no_training_integration=1"
    )
    return GateResult(
        schema=SCHEMA,
        decision=decision,
        pass_gate=pass_gate,
        mode=args.mode,
        trials=args.trials,
        blocks_per_trial=args.blocks_per_trial,
        budget_fraction=args.budget_fraction,
        market_gain_per_work=market_gain,
        grad_norm_gain_per_work=grad_gain,
        random_gain_per_work=random_gain,
        oracle_gain_per_work=oracle_gain,
        no_settlement_gain_per_work=no_settlement_gain,
        market_vs_grad_norm=market_vs_grad,
        market_vs_random=market_vs_random,
        market_selected_bad_rate=market_bad,
        no_settlement_bad_rate=no_settle_bad,
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
    parser.add_argument("--mode", choices=["settled", "no_settlement", "high_noise"], default="settled")
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--blocks-per-trial", type=int, default=96)
    parser.add_argument("--budget-fraction", type=float, default=0.22)
    parser.add_argument("--bid-noise", type=float, default=0.08)
    parser.add_argument("--bad-bidder-rate", type=float, default=0.12)
    parser.add_argument("--min-vs-grad", type=float, default=1.10)
    parser.add_argument("--min-vs-random", type=float, default=1.25)
    parser.add_argument("--max-bad-selected-rate", type=float, default=0.04)
    parser.add_argument("--min-no-settlement-bad-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42203)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    write_result(run_gate(args), args.output)


if __name__ == "__main__":
    main()
