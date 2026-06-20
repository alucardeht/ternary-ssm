"""R440 curriculum generation and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

from .core import ConstructiveCognitiveCore, ConstructiveCoreConfig
from .metrics import profile_constructive_core

R440Stage = Literal[
    "visible_position",
    "visible_transition",
    "occluded_position",
    "occluded_transition",
    "mixed",
]

R440_CURRICULUM_V1: tuple[R440Stage, ...] = (
    "visible_position",
    "visible_transition",
    "occluded_position",
    "occluded_transition",
    "mixed",
)


@dataclass
class R440Batch:
    frame1: torch.Tensor
    frame2: torch.Tensor
    extra: torch.Tensor
    target_color: torch.Tensor
    operator: torch.Tensor
    hard: torch.Tensor
    targets: dict[str, torch.Tensor]
    stage: str


@dataclass(frozen=True)
class R440CurriculumResult:
    seed: int
    steps_per_stage: int
    stages: tuple[str, ...]
    passed: bool
    metrics: dict[str, dict[str, float]]
    cost_profile: dict[str, float | int]


@dataclass(frozen=True)
class R440SeedGridResult:
    seeds: tuple[int, ...]
    passed: bool
    results: dict[int, R440CurriculumResult]


def one_hot(index: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(index.long(), num_classes=classes).float()


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


class R440Curriculum:
    def __init__(
        self,
        *,
        seed: int,
        batch_size: int = 128,
        config: ConstructiveCoreConfig | None = None,
    ):
        torch.manual_seed(seed)
        self.batch_size = batch_size
        self.config = config or ConstructiveCoreConfig()
        grid = self.config.grid
        cells = self.config.cells
        self.coord = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, grid),
                torch.linspace(-1.0, 1.0, grid),
                indexing="ij",
            ),
            dim=-1,
        ).view(cells, 2)

    def shift(self, pos: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        grid = self.config.grid
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), direction]
        return selected[:, 0] * grid + selected[:, 1]

    def sample_positions(self) -> torch.Tensor:
        return torch.stack([torch.randperm(self.config.cells)[: self.config.colors] for _ in range(self.batch_size)])

    def sample_valid_actions(self, pos: torch.Tensor) -> torch.Tensor:
        grid = self.config.grid
        row = pos // grid
        col = pos % grid
        valid = torch.stack((row > 0, row < grid - 1, col > 0, col < grid - 1), dim=-1).float()
        return torch.multinomial(valid / valid.sum(dim=-1, keepdim=True), 1).squeeze(-1)

    def make_frame(self, positions: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        frame = torch.zeros(self.batch_size, cfg.frame_dim, cfg.cells)
        for color in range(cfg.colors):
            visible = visible_mask[:, color]
            frame[visible, color, positions[visible, color]] = 1.0
        frame[:, cfg.colors :, :] = self.coord.T.unsqueeze(0).expand(self.batch_size, 2, cfg.cells)
        return frame.transpose(1, 2)

    def make_batch(self, stage: R440Stage = "mixed") -> R440Batch:
        cfg = self.config
        positions = self.sample_positions()
        target_color = torch.randint(0, cfg.colors, (self.batch_size,))
        target_pos = positions[torch.arange(self.batch_size), target_color]
        action = self.sample_valid_actions(target_pos)

        if stage == "visible_position":
            operator = torch.zeros(self.batch_size, dtype=torch.long)
            hard = torch.zeros(self.batch_size, dtype=torch.bool)
        elif stage == "visible_transition":
            operator = torch.ones(self.batch_size, dtype=torch.long)
            hard = torch.zeros(self.batch_size, dtype=torch.bool)
        elif stage == "occluded_position":
            operator = torch.zeros(self.batch_size, dtype=torch.long)
            hard = torch.ones(self.batch_size, dtype=torch.bool)
        elif stage == "occluded_transition":
            operator = torch.ones(self.batch_size, dtype=torch.long)
            hard = torch.ones(self.batch_size, dtype=torch.bool)
        else:
            operator = torch.randint(0, cfg.operators, (self.batch_size,))
            hard = torch.rand(self.batch_size) < 0.5

        visible_all = torch.ones(self.batch_size, cfg.colors, dtype=torch.bool)
        current_visible = visible_all.clone()
        current_visible[hard, target_color[hard]] = False
        frame1 = self.make_frame(positions, visible_all)
        frame2 = self.make_frame(positions, current_visible)
        extra = torch.cat(
            (one_hot(target_color, cfg.colors), one_hot(operator, cfg.operators), one_hot(action, cfg.actions)),
            dim=-1,
        )
        next_pos = self.shift(target_pos, action)
        answer = torch.where(operator == 0, target_pos, next_pos)
        return R440Batch(
            frame1=frame1,
            frame2=frame2,
            extra=extra,
            target_color=target_color,
            operator=operator,
            hard=hard,
            targets={
                "answer": answer,
                "depth_gate": hard.long(),
                "target_slot": target_pos,
            },
            stage=stage,
        )

    def make_no_image_batch(self, stage: R440Stage = "mixed") -> R440Batch:
        batch = self.make_batch(stage)
        blank1 = torch.zeros_like(batch.frame1)
        blank2 = torch.zeros_like(batch.frame2)
        blank1[:, :, self.config.colors :] = self.coord.unsqueeze(0).expand(self.batch_size, self.config.cells, 2)
        blank2[:, :, self.config.colors :] = self.coord.unsqueeze(0).expand(self.batch_size, self.config.cells, 2)
        return R440Batch(
            frame1=blank1,
            frame2=blank2,
            extra=batch.extra,
            target_color=batch.target_color,
            operator=batch.operator,
            hard=batch.hard,
            targets=batch.targets,
            stage=stage,
        )


def r440_loss(outputs, targets: dict[str, torch.Tensor], hard: torch.Tensor) -> torch.Tensor:
    answer_loss = F.cross_entropy(outputs.answer, targets["answer"])
    deep_loss = F.cross_entropy(outputs.deep_answer, targets["answer"])
    gate_loss = F.binary_cross_entropy_with_logits(outputs.depth_gate, hard.float())
    deep_slot_loss = F.nll_loss(torch.log(outputs.deep_attention + 1e-6), targets["target_slot"])
    easy = ~hard
    if bool(easy.any()):
        fast_loss = F.cross_entropy(outputs.fast_answer[easy], targets["answer"][easy])
        fast_slot_loss = F.nll_loss(torch.log(outputs.fast_attention[easy] + 1e-6), targets["target_slot"][easy])
    else:
        fast_loss = torch.tensor(0.0)
        fast_slot_loss = torch.tensor(0.0)
    return answer_loss + deep_loss + 2.5 * fast_loss + 2.0 * gate_loss + deep_slot_loss + fast_slot_loss


def r440_fast_rehearsal_loss(outputs, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    fast_answer_loss = F.cross_entropy(outputs.fast_answer, targets["answer"])
    fast_slot_loss = F.nll_loss(torch.log(outputs.fast_attention + 1e-6), targets["target_slot"])
    return fast_answer_loss + fast_slot_loss


def evaluate_r440_core(
    model: ConstructiveCognitiveCore,
    curriculum: R440Curriculum,
    *,
    stage: R440Stage = "mixed",
) -> dict[str, float]:
    batch = curriculum.make_batch(stage)
    outputs = model(batch.frame1, batch.frame2, batch.extra)
    gate_pred = (outputs.depth_gate > 0).long()
    easy = ~batch.hard
    hard = batch.hard
    metrics = {
        "answer": accuracy(outputs.answer, batch.targets["answer"]),
        "depth_gate": float((gate_pred == batch.targets["depth_gate"]).float().mean().item()),
        "route_probability": float(outputs.route_rate.item()),
        "depth_separation": 0.0,
        "fast_easy_answer": 0.0,
        "deep_hard_answer": 0.0,
    }
    if bool(easy.any()):
        metrics["fast_easy_answer"] = accuracy(outputs.fast_answer[easy], batch.targets["answer"][easy])
    if bool(hard.any()):
        metrics["deep_hard_answer"] = accuracy(outputs.deep_answer[hard], batch.targets["answer"][hard])
    if bool(easy.any()) and bool(hard.any()):
        metrics["depth_separation"] = float(
            outputs.depth_probability[hard].mean().item() - outputs.depth_probability[easy].mean().item()
        )

    no_image = curriculum.make_no_image_batch(stage)
    metrics["no_image_answer"] = accuracy(
        model(no_image.frame1, no_image.frame2, no_image.extra).answer,
        no_image.targets["answer"],
    )
    metrics["slot_ablation_answer"] = accuracy(
        model(batch.frame1, batch.frame2, batch.extra, ablate_slots=True).answer,
        batch.targets["answer"],
    )

    swapped_extra = batch.extra.clone()
    swapped_color = (
        batch.target_color + 1 + torch.randint(0, curriculum.config.colors - 1, (curriculum.batch_size,))
    ) % curriculum.config.colors
    swapped_extra[:, : curriculum.config.colors] = one_hot(swapped_color, curriculum.config.colors)
    metrics["query_swap_answer"] = accuracy(
        model(batch.frame1, batch.frame2, swapped_extra).answer,
        batch.targets["answer"],
    )

    shuffled_extra = batch.extra.clone()
    shuffled_extra[:, curriculum.config.colors : curriculum.config.colors + curriculum.config.operators] = one_hot(
        1 - batch.operator,
        curriculum.config.operators,
    )
    metrics["proposition_shuffle_answer"] = accuracy(
        model(batch.frame1, batch.frame2, shuffled_extra).answer,
        batch.targets["answer"],
    )
    if bool(hard.any()):
        metrics["force_fast_hard_answer"] = accuracy(
            model(batch.frame1, batch.frame2, batch.extra, force_fast=True).answer[hard],
            batch.targets["answer"][hard],
        )
    else:
        metrics["force_fast_hard_answer"] = 0.0
    return metrics


def stage_passed(stage: R440Stage, metrics: dict[str, float]) -> bool:
    if metrics["answer"] < 0.90:
        return False
    if metrics["no_image_answer"] > 0.30 or metrics["slot_ablation_answer"] > 0.35:
        return False
    if metrics["query_swap_answer"] > 0.35 or metrics["proposition_shuffle_answer"] > 0.35:
        return False
    if stage.startswith("visible"):
        return metrics["fast_easy_answer"] >= 0.85 and metrics["route_probability"] <= 0.35
    if stage.startswith("occluded"):
        return metrics["deep_hard_answer"] >= 0.85 and metrics["route_probability"] >= 0.65
    return (
        metrics["depth_gate"] >= 0.90
        and metrics["fast_easy_answer"] >= 0.85
        and metrics["deep_hard_answer"] >= 0.85
        and metrics["depth_separation"] >= 0.50
        and metrics["force_fast_hard_answer"] <= 0.35
    )


def train_r440_curriculum(
    *,
    seed: int = 101,
    batch_size: int = 128,
    steps_per_stage: int = 320,
    stages: tuple[R440Stage, ...] = R440_CURRICULUM_V1,
    lr: float = 0.006,
    visible_rehearsal_weight: float = 0.35,
    visible_rehearsal_interval: int = 2,
) -> tuple[ConstructiveCognitiveCore, R440Curriculum, R440CurriculumResult]:
    curriculum = R440Curriculum(seed=seed, batch_size=batch_size)
    model = ConstructiveCognitiveCore(curriculum.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for stage in stages:
        for step in range(steps_per_stage):
            batch = curriculum.make_batch(stage)
            loss = r440_loss(model(batch.frame1, batch.frame2, batch.extra), batch.targets, batch.hard)
            if (
                visible_rehearsal_weight > 0.0
                and visible_rehearsal_interval > 0
                and step % visible_rehearsal_interval == 0
            ):
                rehearsal_stage: R440Stage = "visible_position" if step % (visible_rehearsal_interval * 2) == 0 else "visible_transition"
                rehearsal = curriculum.make_batch(rehearsal_stage)
                rehearsal_loss = r440_fast_rehearsal_loss(
                    model(rehearsal.frame1, rehearsal.frame2, rehearsal.extra),
                    rehearsal.targets,
                )
                loss = loss + visible_rehearsal_weight * rehearsal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    metrics = {stage: evaluate_r440_core(model, curriculum, stage=stage) for stage in stages}
    metrics["heldout_mixed"] = evaluate_r440_core(model, R440Curriculum(seed=seed + 1000, batch_size=batch_size), stage="mixed")
    passed = all(stage_passed(stage, metrics[stage]) for stage in stages) and stage_passed("mixed", metrics["heldout_mixed"])
    profile = profile_constructive_core(
        model,
        route_probability=metrics["mixed"]["route_probability"] if "mixed" in metrics else metrics["heldout_mixed"]["route_probability"],
    )
    result = R440CurriculumResult(
        seed=seed,
        steps_per_stage=steps_per_stage,
        stages=tuple(stages),
        passed=passed,
        metrics=metrics,
        cost_profile=profile.as_dict(),
    )
    return model, curriculum, result


def evaluate_r440_seed_grid(
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808),
    *,
    batch_size: int = 128,
    steps_per_stage: int = 320,
    visible_rehearsal_weight: float = 0.35,
    visible_rehearsal_interval: int = 2,
) -> R440SeedGridResult:
    results: dict[int, R440CurriculumResult] = {}
    for seed in seeds:
        _model, _curriculum, result = train_r440_curriculum(
            seed=seed,
            batch_size=batch_size,
            steps_per_stage=steps_per_stage,
            visible_rehearsal_weight=visible_rehearsal_weight,
            visible_rehearsal_interval=visible_rehearsal_interval,
        )
        results[seed] = result
    return R440SeedGridResult(
        seeds=tuple(seeds),
        passed=all(result.passed for result in results.values()),
        results=results,
    )
