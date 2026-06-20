#!/usr/bin/env python3
"""R425 conventional Transformer update-object empirical gate.

This is a small, conventional decoder-only Transformer gate for the R425 paper
thesis: previous training intervals can emit reusable update objects that buy
durable CE progress more efficiently than replaying tokens, selected examples,
distillation, or ordinary checkpoint reuse.

The script is intentionally conservative:

- all source-derived arms start from the same `theta_t`;
- source batches, continuation batches, and evaluation batches are fixed per
  seed;
- object arms are compared against raw replay, gradient-alignment selected
  replay, checkpoint distillation, warm-start, no-source continuation, and
  shuffled capsule ablations;
- reported "paid units" are token-backward-equivalent accounting proxies, not
  claimed hardware FLOPs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters


DATASET_MAGIC = 0x544F4B4E
DATASET_HEADER_BYTES = 16
DEFAULT_VOCAB = 32064
SCHEMA = "r425_update_objects_empirical_gate.v1"


@dataclass
class ArmResult:
    seed: int
    arm: str
    status: str
    initial_eval_ce: float
    post_replay_eval_ce: float
    final_eval_ce: float
    durable_delta_ce: float
    paid_units: float
    durable_delta_ce_per_paid_unit: float
    old_slice_ce: float
    future_slice_ce: float
    retention_delta_vs_raw: float
    replay_steps: int
    continuation_steps: int
    build_units: float
    apply_units: float
    selection_units: float
    distill_units: float
    object_density: float
    elapsed_seconds: float


@dataclass
class GateSummary:
    schema: str
    decision: str
    status: str
    dataset: str
    device: str
    seeds: list[int]
    n_tokens: int
    vocab_size: int
    context_len: int
    batch_size: int
    d_model: int
    n_layers: int
    n_heads: int
    source_steps: int
    replay_steps: int
    continuation_steps: int
    eval_batches: int
    compression_fraction: float
    best_object_arm: str
    best_object_mean: float
    best_control_arm: str
    best_control_mean: float
    object_vs_best_control_ratio: float
    object_wins_by_seed: int
    total_seeds: int
    warm_start_mean: float
    shuffled_capsule_mean: float
    raw_replay_mean: float
    selected_replay_mean: float
    distillation_mean: float
    no_source_continuation_mean: float
    elapsed_seconds: float
    activation_log: str
    notes: list[str]
    arms: list[ArmResult]


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq = x.shape
        pos = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, seq)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        h = self.blocks(h, mask=mask)
        return self.lm_head(self.norm(h))


def read_tokens(path: Path, max_tokens: int | None) -> tuple[torch.Tensor, int]:
    data = path.read_bytes()
    offset = 0
    n_tokens = len(data) // 2
    vocab_size = DEFAULT_VOCAB
    if len(data) >= DATASET_HEADER_BYTES:
        magic, version, header_tokens = struct.unpack_from("<IIQ", data, 0)
        if magic == DATASET_MAGIC:
            if version != 1:
                raise ValueError(f"unsupported dataset version: {version}")
            offset = DATASET_HEADER_BYTES
            n_tokens = int(header_tokens)
    raw = np.frombuffer(data, dtype="<u2", count=n_tokens, offset=offset).astype(np.int64)
    tokens = torch.from_numpy(raw).to(torch.long)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    if tokens.numel() > 0:
        vocab_size = max(DEFAULT_VOCAB, int(tokens.max().item()) + 1)
    return tokens.contiguous(), vocab_size


def make_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> TinyDecoderLM:
    model = TinyDecoderLM(
        vocab_size=vocab_size,
        context_len=args.context_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    return model


def split_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = tokens.numel()
    train_end = int(n * 0.70)
    eval_end = int(n * 0.85)
    return tokens[:train_end], tokens[train_end:eval_end], tokens[eval_end:]


def sample_starts(
    rng: random.Random,
    token_count: int,
    count: int,
    context_len: int,
    batch_size: int,
) -> list[list[int]]:
    max_start = token_count - context_len - 2
    if max_start <= 0:
        raise ValueError("dataset split is too small for context_len")
    return [
        [rng.randint(0, max_start) for _ in range(batch_size)]
        for _ in range(count)
    ]


def batch_from_starts(
    tokens: torch.Tensor,
    starts: list[int],
    context_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = [tokens[s : s + context_len] for s in starts]
    ys = [tokens[s + 1 : s + context_len + 1] for s in starts]
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def ce_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokens: torch.Tensor,
    starts: list[list[int]],
    context_len: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_starts in starts:
        x, y = batch_from_starts(tokens, batch_starts, context_len, device)
        losses.append(float(ce_loss(model, x, y).detach().cpu()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def train_ce_steps(
    model: nn.Module,
    tokens: torch.Tensor,
    batches: list[list[int]],
    context_len: int,
    device: torch.device,
    lr: float,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    for starts in batches:
        x, y = batch_from_starts(tokens, starts, context_len, device)
        opt.zero_grad(set_to_none=True)
        loss = ce_loss(model, x, y)
        loss.backward()
        opt.step()


def grad_vector(model: nn.Module) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(chunks)


def apply_vector_delta(model: nn.Module, delta: torch.Tensor) -> None:
    with torch.no_grad():
        vec = parameters_to_vector(model.parameters())
        vector_to_parameters(vec + delta.to(vec.device), model.parameters())


def topk_compress(vec: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = vec.detach().flatten()
    keep = max(1, int(flat.numel() * fraction))
    if keep >= flat.numel():
        return flat.clone()
    _, idx = torch.topk(flat.abs(), keep, sorted=False)
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    return out


def topk_select_by_score(values: torch.Tensor, scores: torch.Tensor, fraction: float) -> torch.Tensor:
    flat_values = values.detach().flatten()
    flat_scores = scores.detach().flatten().abs()
    keep = max(1, int(flat_values.numel() * fraction))
    if keep >= flat_values.numel():
        return flat_values.clone()
    _, idx = torch.topk(flat_scores, keep, sorted=False)
    out = torch.zeros_like(flat_values)
    out[idx] = flat_values[idx]
    return out


def parameter_block_sizes(model: nn.Module) -> list[int]:
    return [p.numel() for p in model.parameters()]


def topk_compress_by_blocks(vec: torch.Tensor, fraction: float, block_sizes: list[int]) -> torch.Tensor:
    flat = vec.detach().flatten()
    out = torch.zeros_like(flat)
    offset = 0
    for size in block_sizes:
        block = flat[offset : offset + size]
        if block.numel():
            keep = max(1, int(block.numel() * fraction))
            keep = min(keep, block.numel())
            _, idx = torch.topk(block.abs(), keep, sorted=False)
            out[offset + idx] = block[idx]
        offset += size
    return out


def topk_select_by_score_by_blocks(
    values: torch.Tensor,
    scores: torch.Tensor,
    fraction: float,
    block_sizes: list[int],
) -> torch.Tensor:
    flat_values = values.detach().flatten()
    flat_scores = scores.detach().flatten().abs()
    out = torch.zeros_like(flat_values)
    offset = 0
    for size in block_sizes:
        value_block = flat_values[offset : offset + size]
        score_block = flat_scores[offset : offset + size]
        if value_block.numel():
            keep = max(1, int(value_block.numel() * fraction))
            keep = min(keep, value_block.numel())
            _, idx = torch.topk(score_block, keep, sorted=False)
            out[offset + idx] = value_block[idx]
        offset += size
    return out


def alignment_weighted_gradient_capsule(
    per_batch_grads: list[torch.Tensor],
    alignments: list[float],
    fraction: float,
) -> torch.Tensor:
    stack = torch.stack(per_batch_grads)
    weights = torch.tensor(alignments, dtype=stack.dtype).clamp_min(0.0)
    if float(weights.sum()) <= 1e-12:
        weights = torch.ones_like(weights)
    weighted_grad = (stack * (weights / weights.sum()).unsqueeze(1)).sum(dim=0)
    return topk_compress(weighted_grad, fraction)


def sign_consensus_gradient_capsule(per_batch_grads: list[torch.Tensor], fraction: float) -> torch.Tensor:
    stack = torch.stack(per_batch_grads)
    mean_grad = stack.mean(dim=0)
    consensus = stack.sign().mean(dim=0).abs()
    score = mean_grad.abs() * consensus
    return topk_select_by_score(mean_grad, score, fraction)


def optimizer_transaction_capsule(
    per_step_deltas: list[torch.Tensor],
    fraction: float,
    block_sizes: list[int],
) -> torch.Tensor:
    stack = torch.stack(per_step_deltas)
    summed_update = stack.sum(dim=0)
    cumulative_movement = stack.abs().sum(dim=0)
    return topk_select_by_score_by_blocks(summed_update, cumulative_movement, fraction, block_sizes)


def shuffled_like(vec: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=vec.device)
    generator.manual_seed(seed)
    idx = torch.randperm(vec.numel(), generator=generator, device=vec.device)
    return vec[idx]


def compute_source_interval(
    args: argparse.Namespace,
    vocab_size: int,
    train_tokens: torch.Tensor,
    source_batches: list[list[int]],
    eval_batches: list[list[int]],
    seed: int,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[float],
    list[torch.Tensor],
    list[torch.Tensor],
    list[float],
]:
    torch.manual_seed(seed)
    model = make_model(args, vocab_size, device)
    theta_t_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    per_batch_grads: list[torch.Tensor] = []
    per_step_deltas: list[torch.Tensor] = []
    per_batch_losses: list[float] = []
    for starts in source_batches:
        x, y = batch_from_starts(train_tokens, starts, args.context_len, device)
        opt.zero_grad(set_to_none=True)
        loss = ce_loss(model, x, y)
        loss.backward()
        per_batch_grads.append(grad_vector(model).detach().cpu())
        per_batch_losses.append(float(loss.detach().cpu()))
        before_step = parameters_to_vector(model.parameters()).detach().cpu().clone()
        opt.step()
        after_step = parameters_to_vector(model.parameters()).detach().cpu().clone()
        per_step_deltas.append(after_step - before_step)

    theta_tk_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    opt.zero_grad(set_to_none=True)
    dev_starts = eval_batches[0]
    x_dev, y_dev = batch_from_starts(train_tokens, dev_starts, args.context_len, device)
    dev_loss = ce_loss(model, x_dev, y_dev)
    dev_loss.backward()
    dev_grad = grad_vector(model).detach().cpu()

    alignments = [
        float(torch.dot(g, dev_grad) / ((g.norm() * dev_grad.norm()).clamp_min(1e-12)))
        for g in per_batch_grads
    ]
    return theta_t_state, theta_tk_state, alignments, per_batch_grads, per_step_deltas, per_batch_losses


def load_model_from_state(
    args: argparse.Namespace,
    vocab_size: int,
    device: torch.device,
    state: dict[str, torch.Tensor],
) -> TinyDecoderLM:
    model = make_model(args, vocab_size, device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    return model


def distill_steps(
    student: nn.Module,
    teacher: nn.Module,
    tokens: torch.Tensor,
    batches: list[list[int]],
    context_len: int,
    device: torch.device,
    lr: float,
    temperature: float,
) -> None:
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    teacher.eval()
    for starts in batches:
        x, _ = batch_from_starts(tokens, starts, context_len, device)
        with torch.no_grad():
            teacher_logp = F.log_softmax(teacher(x) / temperature, dim=-1)
        opt.zero_grad(set_to_none=True)
        student_logp = F.log_softmax(student(x) / temperature, dim=-1)
        loss = F.kl_div(student_logp, teacher_logp, log_target=True, reduction="batchmean") * (temperature**2)
        loss.backward()
        opt.step()


def run_arm(
    args: argparse.Namespace,
    arm: str,
    seed: int,
    vocab_size: int,
    train_tokens: torch.Tensor,
    eval_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    theta_t_state: dict[str, torch.Tensor],
    theta_tk_state: dict[str, torch.Tensor],
    source_batches: list[list[int]],
    continuation_batches: list[list[int]],
    eval_batches: list[list[int]],
    old_eval_batches: list[list[int]],
    future_eval_batches: list[list[int]],
    selected_indices: list[int],
    capsule_objects: dict[str, torch.Tensor],
    delta_object: torch.Tensor,
    initial_eval_ce: float,
    device: torch.device,
) -> ArmResult:
    start = time.perf_counter()
    build_units = 0.0
    apply_units = 0.0
    selection_units = 0.0
    distill_units = 0.0
    object_density = 0.0

    model = load_model_from_state(args, vocab_size, device, theta_t_state)
    replay_batches = source_batches[: args.replay_steps]

    if arm == "raw_replay":
        train_ce_steps(model, train_tokens, replay_batches, args.context_len, device, args.lr)
    elif arm == "selected_replay":
        selection_units = args.source_steps * args.selection_cost_per_source_step
        selected_batches = [source_batches[i] for i in selected_indices[: args.replay_steps]]
        if not selected_batches:
            selected_batches = replay_batches
        while len(selected_batches) < args.replay_steps:
            selected_batches.extend(selected_batches)
        train_ce_steps(model, train_tokens, selected_batches[: args.replay_steps], args.context_len, device, args.lr)
    elif arm == "distillation":
        distill_units = args.replay_steps * args.distill_teacher_cost_per_step
        teacher = load_model_from_state(args, vocab_size, device, theta_tk_state)
        distill_steps(
            model,
            teacher,
            train_tokens,
            replay_batches,
            args.context_len,
            device,
            args.lr,
            args.distill_temperature,
        )
    elif arm == "checkpoint_delta_object":
        build_units = args.delta_build_units
        apply_units = args.delta_apply_units
        object_density = float((delta_object != 0).float().mean().item())
        apply_vector_delta(model, delta_object.to(device) * args.delta_scale)
    elif arm == "shuffled_delta_ablation":
        build_units = args.delta_build_units
        apply_units = args.delta_apply_units
        shuffled = shuffled_like(delta_object, seed + 420_425)
        object_density = float((shuffled != 0).float().mean().item())
        apply_vector_delta(model, shuffled.to(device) * args.delta_scale)
    elif arm in capsule_objects:
        build_units = args.capsule_build_units
        apply_units = args.capsule_apply_units
        capsule = capsule_objects[arm]
        object_density = float((capsule != 0).float().mean().item())
        apply_vector_delta(model, args.capsule_scale * capsule.to(device))
    elif arm.startswith("shuffled_") and arm.removeprefix("shuffled_") in capsule_objects:
        build_units = args.capsule_build_units
        apply_units = args.capsule_apply_units
        capsule = capsule_objects[arm.removeprefix("shuffled_")]
        shuffled = shuffled_like(capsule, seed + 910_425)
        object_density = float((shuffled != 0).float().mean().item())
        apply_vector_delta(model, args.capsule_scale * shuffled.to(device))
    elif arm == "no_source_continuation":
        pass
    elif arm == "warm_start_checkpoint":
        model = load_model_from_state(args, vocab_size, device, theta_tk_state)
        build_units = args.warm_start_build_units
        apply_units = args.warm_start_apply_units
        object_density = 1.0
    else:
        raise ValueError(f"unknown arm: {arm}")

    post_replay_eval_ce = evaluate(model, eval_tokens, eval_batches, args.context_len, device)

    if args.continuation_steps:
        train_ce_steps(
            model,
            train_tokens,
            continuation_batches[: args.continuation_steps],
            args.context_len,
            device,
            args.lr,
        )
    final_eval_ce = evaluate(model, eval_tokens, eval_batches, args.context_len, device)
    old_slice_ce = evaluate(model, eval_tokens, old_eval_batches, args.context_len, device)
    future_slice_ce = evaluate(model, future_tokens, future_eval_batches, args.context_len, device)

    token_units = 0.0
    if arm in {"raw_replay", "selected_replay", "distillation"}:
        token_units += args.replay_steps
    token_units += args.continuation_steps
    paid_units = token_units + build_units + apply_units + selection_units + distill_units
    durable_delta = initial_eval_ce - final_eval_ce
    score = durable_delta / max(1e-12, paid_units)
    elapsed = time.perf_counter() - start

    return ArmResult(
        seed=seed,
        arm=arm,
        status="ok",
        initial_eval_ce=initial_eval_ce,
        post_replay_eval_ce=post_replay_eval_ce,
        final_eval_ce=final_eval_ce,
        durable_delta_ce=durable_delta,
        paid_units=paid_units,
        durable_delta_ce_per_paid_unit=score,
        old_slice_ce=old_slice_ce,
        future_slice_ce=future_slice_ce,
        retention_delta_vs_raw=0.0,
        replay_steps=args.replay_steps,
        continuation_steps=args.continuation_steps,
        build_units=build_units,
        apply_units=apply_units,
        selection_units=selection_units,
        distill_units=distill_units,
        object_density=object_density,
        elapsed_seconds=elapsed,
    )


def mean_metric(results: Iterable[ArmResult], arm: str) -> float:
    vals = [r.durable_delta_ce_per_paid_unit for r in results if r.arm == arm]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def write_markdown(summary: GateSummary, path: Path) -> None:
    by_seed: dict[int, list[ArmResult]] = {}
    for result in summary.arms:
        by_seed.setdefault(result.seed, []).append(result)

    lines: list[str] = []
    lines.append("# R425 Update Objects Empirical Gate")
    lines.append("")
    lines.append(f"Schema: `{summary.schema}`")
    lines.append(f"Decision: `{summary.decision}`")
    lines.append(f"Status: `{summary.status}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Dataset: `{summary.dataset}`")
    lines.append(f"- Device: `{summary.device}`")
    lines.append(f"- Seeds: `{', '.join(map(str, summary.seeds))}`")
    lines.append(f"- Tokens: `{summary.n_tokens}`")
    lines.append(f"- Vocab size: `{summary.vocab_size}`")
    lines.append(f"- Transformer: d_model `{summary.d_model}`, layers `{summary.n_layers}`, heads `{summary.n_heads}`")
    lines.append(f"- Source/replay/continuation steps: `{summary.source_steps}/{summary.replay_steps}/{summary.continuation_steps}`")
    lines.append("")
    lines.append("## Mean Paid Durable CE")
    lines.append("")
    lines.append("| arm | mean durable delta CE / paid unit |")
    lines.append("|---|---:|")
    for arm in sorted({r.arm for r in summary.arms}):
        lines.append(f"| `{arm}` | {mean_metric(summary.arms, arm):.8f} |")
    lines.append("")
    lines.append("## Gate Comparison")
    lines.append("")
    lines.append(f"- Best object arm: `{summary.best_object_arm}` = `{summary.best_object_mean:.8f}`")
    lines.append(f"- Best non-object control: `{summary.best_control_arm}` = `{summary.best_control_mean:.8f}`")
    lines.append(f"- Object/control ratio: `{summary.object_vs_best_control_ratio:.6f}`")
    lines.append(f"- Object wins by seed: `{summary.object_wins_by_seed}/{summary.total_seeds}`")
    lines.append(f"- Warm-start mean: `{summary.warm_start_mean:.8f}`")
    lines.append(f"- Shuffled capsule ablation mean: `{summary.shuffled_capsule_mean:.8f}`")
    lines.append("")
    lines.append("## Per-Seed Results")
    lines.append("")
    lines.append("| seed | arm | final CE | durable delta | paid units | score | old CE | future CE | density |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for seed in sorted(by_seed):
        for r in sorted(by_seed[seed], key=lambda x: x.arm):
            lines.append(
                f"| {seed} | `{r.arm}` | {r.final_eval_ce:.6f} | {r.durable_delta_ce:.6f} | "
                f"{r.paid_units:.3f} | {r.durable_delta_ce_per_paid_unit:.8f} | "
                f"{r.old_slice_ce:.6f} | {r.future_slice_ce:.6f} | {r.object_density:.6f} |"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in summary.notes:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("src/model_v12/wiki_1mb.bin"))
    parser.add_argument("--output", type=Path, default=Path(".workspace/research/r425-conventional-high-impact-paper/empirical-gate-result.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path(".workspace/research/r425-conventional-high-impact-paper/empirical-gate-result.md"))
    parser.add_argument("--max-tokens", type=int, default=80000)
    parser.add_argument("--seeds", type=str, default="4250,4251,4252")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--source-steps", type=int, default=24)
    parser.add_argument("--replay-steps", type=int, default=12)
    parser.add_argument("--continuation-steps", type=int, default=12)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--compression-fraction", type=float, default=0.02)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--capsule-scale", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--selection-cost-per-source-step", type=float, default=0.05)
    parser.add_argument("--distill-teacher-cost-per-step", type=float, default=0.35)
    parser.add_argument("--delta-build-units", type=float, default=1.0)
    parser.add_argument("--delta-apply-units", type=float, default=0.2)
    parser.add_argument("--capsule-build-units", type=float, default=1.5)
    parser.add_argument("--capsule-apply-units", type=float, default=0.4)
    parser.add_argument("--warm-start-build-units", type=float, default=24.0)
    parser.add_argument("--warm-start-apply-units", type=float, default=0.5)
    parser.add_argument("--paper-threshold", type=float, default=1.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    tokens, vocab_size = read_tokens(args.dataset, args.max_tokens)
    train_tokens, eval_tokens, future_tokens = split_tokens(tokens)

    all_results: list[ArmResult] = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        rng = random.Random(seed)

        source_batches = sample_starts(rng, train_tokens.numel(), args.source_steps, args.context_len, args.batch_size)
        continuation_batches = sample_starts(rng, train_tokens.numel(), args.continuation_steps, args.context_len, args.batch_size)
        eval_batches = sample_starts(rng, eval_tokens.numel(), args.eval_batches, args.context_len, args.batch_size)
        old_eval_batches = sample_starts(rng, eval_tokens.numel(), max(4, args.eval_batches // 2), args.context_len, args.batch_size)
        future_eval_batches = sample_starts(rng, future_tokens.numel(), max(4, args.eval_batches // 2), args.context_len, args.batch_size)

        theta_t, theta_tk, alignments, per_batch_grads, per_step_deltas, _ = compute_source_interval(
            args,
            vocab_size,
            train_tokens,
            source_batches,
            eval_batches,
            seed,
            device,
        )

        initial_model = load_model_from_state(args, vocab_size, device, theta_t)
        initial_eval_ce = evaluate(initial_model, eval_tokens, eval_batches, args.context_len, device)

        sorted_indices = sorted(range(len(alignments)), key=lambda i: alignments[i], reverse=True)
        theta_t_model = load_model_from_state(args, vocab_size, torch.device("cpu"), theta_t)
        theta_tk_model = load_model_from_state(args, vocab_size, torch.device("cpu"), theta_tk)
        block_sizes = parameter_block_sizes(theta_t_model)

        grad_mean = torch.stack(per_batch_grads).mean(dim=0)
        capsule_objects = {
            "gradient_transaction_capsule": -args.lr * topk_compress(grad_mean, args.compression_fraction),
            "layerwise_gradient_transaction_capsule": -args.lr
            * topk_compress_by_blocks(grad_mean, args.compression_fraction, block_sizes),
            "alignment_weighted_gradient_capsule": -args.lr
            * alignment_weighted_gradient_capsule(per_batch_grads, alignments, args.compression_fraction),
            "sign_consensus_gradient_capsule": -args.lr
            * sign_consensus_gradient_capsule(per_batch_grads, args.compression_fraction),
            "optimizer_transaction_capsule": optimizer_transaction_capsule(
                per_step_deltas,
                args.compression_fraction,
                block_sizes,
            ),
        }

        delta_full = parameters_to_vector(theta_tk_model.parameters()) - parameters_to_vector(theta_t_model.parameters())
        delta_object = topk_compress(delta_full, args.compression_fraction)

        arms = [
            "raw_replay",
            "selected_replay",
            "distillation",
            "checkpoint_delta_object",
            "shuffled_delta_ablation",
            "gradient_transaction_capsule",
            "shuffled_gradient_transaction_capsule",
            "layerwise_gradient_transaction_capsule",
            "shuffled_layerwise_gradient_transaction_capsule",
            "alignment_weighted_gradient_capsule",
            "shuffled_alignment_weighted_gradient_capsule",
            "sign_consensus_gradient_capsule",
            "shuffled_sign_consensus_gradient_capsule",
            "optimizer_transaction_capsule",
            "shuffled_optimizer_transaction_capsule",
            "no_source_continuation",
            "warm_start_checkpoint",
        ]
        seed_results = [
            run_arm(
                args,
                arm,
                seed,
                vocab_size,
                train_tokens,
                eval_tokens,
                future_tokens,
                theta_t,
                theta_tk,
                source_batches,
                continuation_batches,
                eval_batches,
                old_eval_batches,
                future_eval_batches,
                sorted_indices,
                capsule_objects,
                delta_object,
                initial_eval_ce,
                device,
            )
            for arm in arms
        ]
        raw_final = next(r.final_eval_ce for r in seed_results if r.arm == "raw_replay")
        for result in seed_results:
            result.retention_delta_vs_raw = result.old_slice_ce - raw_final
        all_results.extend(seed_results)

    capsule_arms = {
        "gradient_transaction_capsule",
        "layerwise_gradient_transaction_capsule",
        "alignment_weighted_gradient_capsule",
        "sign_consensus_gradient_capsule",
        "optimizer_transaction_capsule",
    }
    object_arms = {"checkpoint_delta_object", *capsule_arms}
    control_arms = {
        "raw_replay",
        "selected_replay",
        "distillation",
        "no_source_continuation",
    }
    object_means = {arm: mean_metric(all_results, arm) for arm in object_arms}
    control_means = {arm: mean_metric(all_results, arm) for arm in control_arms}
    best_object_arm, best_object_mean = max(object_means.items(), key=lambda kv: kv[1])
    best_control_arm, best_control_mean = max(control_means.items(), key=lambda kv: kv[1])
    ratio = best_object_mean / max(1e-12, best_control_mean)

    wins = 0
    for seed in seeds:
        seed_object = max(r.durable_delta_ce_per_paid_unit for r in all_results if r.seed == seed and r.arm in object_arms)
        seed_control = max(r.durable_delta_ce_per_paid_unit for r in all_results if r.seed == seed and r.arm in control_arms)
        if seed_object > seed_control:
            wins += 1

    checkpoint_delta_mean = mean_metric(all_results, "checkpoint_delta_object")
    shuffled_delta_mean = mean_metric(all_results, "shuffled_delta_ablation")
    gradient_capsule_mean = mean_metric(all_results, "gradient_transaction_capsule")
    no_source_mean = mean_metric(all_results, "no_source_continuation")
    capsule_support = False
    best_capsule_arm = max(capsule_arms, key=lambda arm: mean_metric(all_results, arm))
    best_capsule_mean = mean_metric(all_results, best_capsule_arm)
    best_capsule_shuffle_mean = mean_metric(all_results, f"shuffled_{best_capsule_arm}")
    capsule_wins = 0
    for seed in seeds:
        seed_capsule = max(
            r.durable_delta_ce_per_paid_unit for r in all_results if r.seed == seed and r.arm in capsule_arms
        )
        seed_control = max(
            r.durable_delta_ce_per_paid_unit for r in all_results if r.seed == seed and r.arm in control_arms
        )
        if seed_capsule > seed_control:
            capsule_wins += 1
    capsule_support = (
        best_capsule_mean >= args.paper_threshold * best_control_mean
        and best_capsule_mean >= 1.10 * best_capsule_shuffle_mean
        and best_capsule_mean > no_source_mean
        and capsule_wins >= math.ceil(0.67 * len(seeds))
    )
    checkpoint_delta_valid = (
        checkpoint_delta_mean >= args.paper_threshold * best_control_mean
        and checkpoint_delta_mean >= args.paper_threshold * shuffled_delta_mean
        and wins >= math.ceil(0.67 * len(seeds))
    )
    notes = [
        "Paid units are token-backward-equivalent accounting proxies, not measured GPU FLOPs.",
        "Warm-start is reported as an upper-bound/collapse control, not an admissible update-object arm.",
        "A paper-strong pass requires the checkpoint-delta object to beat replay controls and shuffled-delta ablation, plus independent mature capsule support.",
        f"Best mature capsule arm: {best_capsule_arm} score={best_capsule_mean:.8f} shuffled={best_capsule_shuffle_mean:.8f} wins={capsule_wins}/{len(seeds)}.",
    ]
    if checkpoint_delta_valid and capsule_support:
        decision = "PASS_PAPER_GATE_INITIAL"
        status = "paper_candidate_empirical_positive_initial"
    elif checkpoint_delta_valid:
        decision = "CHECKPOINT_DELTA_POSITIVE_CAPSULE_WEAK"
        status = "partial_positive_checkpoint_delta_only"
    elif best_object_mean > best_control_mean:
        decision = "WEAK_OBJECT_SIGNAL_NOT_PAPER_STRONG"
        status = "positive_but_not_enough_for_paper"
    else:
        decision = "FAIL_OBJECTS_DO_NOT_BEAT_CONTROLS"
        status = "failed_empirical_gate_current_realization"

    activation_log = (
        "[R425_UPDATE_OBJECTS_EMPIRICAL_GATE] "
        f"active=1 decision={decision} best_object={best_object_arm} "
        f"best_control={best_control_arm} ratio={ratio:.6f} seeds={len(seeds)}"
    )
    print(activation_log)

    summary = GateSummary(
        schema=SCHEMA,
        decision=decision,
        status=status,
        dataset=str(args.dataset),
        device=str(device),
        seeds=seeds,
        n_tokens=int(tokens.numel()),
        vocab_size=vocab_size,
        context_len=args.context_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        source_steps=args.source_steps,
        replay_steps=args.replay_steps,
        continuation_steps=args.continuation_steps,
        eval_batches=args.eval_batches,
        compression_fraction=args.compression_fraction,
        best_object_arm=best_object_arm,
        best_object_mean=best_object_mean,
        best_control_arm=best_control_arm,
        best_control_mean=best_control_mean,
        object_vs_best_control_ratio=ratio,
        object_wins_by_seed=wins,
        total_seeds=len(seeds),
        warm_start_mean=mean_metric(all_results, "warm_start_checkpoint"),
        shuffled_capsule_mean=mean_metric(all_results, f"shuffled_{best_capsule_arm}"),
        raw_replay_mean=mean_metric(all_results, "raw_replay"),
        selected_replay_mean=mean_metric(all_results, "selected_replay"),
        distillation_mean=mean_metric(all_results, "distillation"),
        no_source_continuation_mean=mean_metric(all_results, "no_source_continuation"),
        elapsed_seconds=time.perf_counter() - start,
        activation_log=activation_log,
        notes=notes,
        arms=all_results,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(summary, args.markdown_output)


if __name__ == "__main__":
    main()
