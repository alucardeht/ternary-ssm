"""Controlled room-layout teach-test for the R440 core."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import ConstructiveCognitiveCore, ConstructiveCoreConfig
from .curriculum import R440Curriculum, one_hot

ROOM_WALL = "#"
ROOM_DOOR = "D"
ROOM_EMPTY = "."
ROOM_OBJECTS: tuple[str, ...] = ("R", "G", "B")


@dataclass(frozen=True)
class R440RoomLayoutResult:
    seed: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


@dataclass(frozen=True)
class R440RoomUpdateResult:
    seed: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


@dataclass(frozen=True)
class R440RoomRelationResult:
    seed: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


@dataclass(frozen=True)
class R440RoomSequentialUpdateResult:
    seed: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


class R440RoomLayout:
    def __init__(
        self,
        *,
        seed: int,
        batch_size: int = 64,
        config: ConstructiveCoreConfig | None = None,
    ):
        torch.manual_seed(seed)
        self.seed = seed
        self.batch_size = batch_size
        self.config = config or ConstructiveCoreConfig()
        if self.config.grid != 4 or self.config.colors != 3:
            raise ValueError("R440RoomLayout currently expects a 4x4 grid and three object channels")
        self.curriculum = R440Curriculum(seed=seed, batch_size=batch_size, config=self.config)
        self.interior = torch.tensor([5, 6, 9, 10], dtype=torch.long)
        self.wall = torch.tensor([0, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15], dtype=torch.long)
        self.door = 1

    def sample_positions(self) -> torch.Tensor:
        rows = []
        for _ in range(self.batch_size):
            rows.append(self.interior[torch.randperm(self.interior.numel())[: self.config.colors]])
        return torch.stack(rows)

    def make_room_queries(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = self.sample_positions()
        visible = torch.ones(self.batch_size, self.config.colors, dtype=torch.bool)
        hidden_color = torch.randint(0, self.config.colors, (self.batch_size,))
        visible[torch.arange(self.batch_size), hidden_color] = False
        frame1 = self.curriculum.make_frame(positions, torch.ones_like(visible))
        frame2 = self.curriculum.make_frame(positions, visible)
        frames1 = []
        frames2 = []
        extras = []
        targets = []
        for color in range(self.config.colors):
            color_ids = torch.full((self.batch_size,), color, dtype=torch.long)
            operator = torch.zeros(self.batch_size, dtype=torch.long)
            action = torch.zeros(self.batch_size, dtype=torch.long)
            extra = torch.cat(
                (
                    one_hot(color_ids, self.config.colors),
                    one_hot(operator, self.config.operators),
                    one_hot(action, self.config.actions),
                ),
                dim=-1,
            )
            frames1.append(frame1)
            frames2.append(frame2)
            extras.append(extra)
            targets.append(positions[:, color])
        return (
            torch.cat(frames1, dim=0),
            torch.cat(frames2, dim=0),
            torch.cat(extras, dim=0),
            torch.cat(targets, dim=0),
        )

    def make_update_queries(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = self.sample_positions()
        moved_color = torch.randint(0, self.config.colors, (self.batch_size,))
        updated = initial.clone()
        for row in range(self.batch_size):
            occupied = set(int(cell) for cell in initial[row].tolist())
            candidates = [int(cell) for cell in self.interior.tolist() if int(cell) not in occupied]
            updated[row, moved_color[row]] = candidates[torch.randint(0, len(candidates), (1,)).item()]
        visible = torch.ones(self.batch_size, self.config.colors, dtype=torch.bool)
        initial_frame = self.curriculum.make_frame(initial, visible)
        updated_frame = self.curriculum.make_frame(updated, visible)
        frames1 = []
        updated_frames2 = []
        stale_frames2 = []
        extras = []
        targets = []
        for color in range(self.config.colors):
            color_ids = torch.full((self.batch_size,), color, dtype=torch.long)
            operator = torch.zeros(self.batch_size, dtype=torch.long)
            action = torch.zeros(self.batch_size, dtype=torch.long)
            extra = torch.cat(
                (
                    one_hot(color_ids, self.config.colors),
                    one_hot(operator, self.config.operators),
                    one_hot(action, self.config.actions),
                ),
                dim=-1,
            )
            frames1.append(initial_frame)
            updated_frames2.append(updated_frame)
            stale_frames2.append(initial_frame)
            extras.append(extra)
            targets.append(updated[:, color])
        return (
            torch.cat(frames1, dim=0),
            torch.cat(updated_frames2, dim=0),
            torch.cat(stale_frames2, dim=0),
            torch.cat(extras, dim=0),
            torch.cat(targets, dim=0),
            moved_color,
        )

    def make_sequential_update_queries(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = self.sample_positions()
        first = initial.clone()
        first_color = torch.randint(0, self.config.colors, (self.batch_size,))
        second_color = (first_color + 1 + torch.randint(0, self.config.colors - 1, (self.batch_size,))) % self.config.colors
        for row in range(self.batch_size):
            occupied = set(int(cell) for cell in initial[row].tolist())
            first_candidates = [int(cell) for cell in self.interior.tolist() if int(cell) not in occupied]
            first[row, first_color[row]] = first_candidates[torch.randint(0, len(first_candidates), (1,)).item()]
        second = first.clone()
        for row in range(self.batch_size):
            occupied = set(int(cell) for cell in first[row].tolist())
            second_candidates = [int(cell) for cell in self.interior.tolist() if int(cell) not in occupied]
            second[row, second_color[row]] = second_candidates[torch.randint(0, len(second_candidates), (1,)).item()]

        visible = torch.ones(self.batch_size, self.config.colors, dtype=torch.bool)
        initial_frame = self.curriculum.make_frame(initial, visible)
        first_frame = self.curriculum.make_frame(first, visible)
        second_frame = self.curriculum.make_frame(second, visible)
        frames1 = []
        first_frames2 = []
        second_frames2 = []
        extras = []
        targets = []
        for color in range(self.config.colors):
            color_ids = torch.full((self.batch_size,), color, dtype=torch.long)
            operator = torch.zeros(self.batch_size, dtype=torch.long)
            action = torch.zeros(self.batch_size, dtype=torch.long)
            extra = torch.cat(
                (
                    one_hot(color_ids, self.config.colors),
                    one_hot(operator, self.config.operators),
                    one_hot(action, self.config.actions),
                ),
                dim=-1,
            )
            frames1.append(initial_frame)
            first_frames2.append(first_frame)
            second_frames2.append(second_frame)
            extras.append(extra)
            targets.append(second[:, color])
        return (
            torch.cat(frames1, dim=0),
            torch.cat(first_frames2, dim=0),
            torch.cat(second_frames2, dim=0),
            torch.cat(extras, dim=0),
            torch.cat(targets, dim=0),
            first_color,
            second_color,
        )

    def render(self, positions: torch.Tensor) -> list[str]:
        cells = [ROOM_EMPTY for _ in range(self.config.cells)]
        for idx in self.wall.tolist():
            cells[idx] = ROOM_WALL
        cells[self.door] = ROOM_DOOR
        for color, cell in enumerate(positions.tolist()):
            if 0 <= cell < self.config.cells:
                cells[cell] = ROOM_OBJECTS[color]
        return ["".join(cells[row * self.config.grid : (row + 1) * self.config.grid]) for row in range(self.config.grid)]


def room_layout_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["position_accuracy"] >= 0.95
        and metrics["exact_layout"] >= 0.90
        and metrics["no_overlap"] >= 0.95
        and metrics["avoids_wall"] >= 0.95
        and metrics["no_image_exact_layout"] <= 0.35
        and metrics["slot_ablation_exact_layout"] <= 0.35
    )


def room_update_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["updated_exact_layout"] >= 0.90
        and metrics["intended_object_changed"] >= 0.90
        and metrics["unchanged_objects_preserved"] >= 0.90
        and metrics["updated_no_overlap"] >= 0.95
        and metrics["updated_avoids_wall"] >= 0.95
        and metrics["stale_exact_layout"] <= 0.35
        and metrics["slot_ablation_exact_layout"] <= 0.35
    )


def room_relation_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["relation_accuracy"] >= 0.95
        and metrics["exact_relation_set"] >= 0.90
        and metrics["no_image_exact_relation_set"] <= 0.35
        and metrics["slot_ablation_exact_relation_set"] <= 0.35
    )


def room_sequential_update_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["final_exact_layout"] >= 0.90
        and metrics["first_changed_object_correct"] >= 0.90
        and metrics["second_changed_object_correct"] >= 0.90
        and metrics["unmoved_object_preserved"] >= 0.90
        and metrics["final_no_overlap"] >= 0.95
        and metrics["final_avoids_wall"] >= 0.95
        and metrics["intermediate_stale_exact_layout"] <= 0.35
        and metrics["slot_ablation_exact_layout"] <= 0.35
    )


def room_relation_tensor(positions: torch.Tensor, *, grid: int = 4) -> torch.Tensor:
    rows = positions // grid
    cols = positions % grid
    relation_rows = []
    for source in range(positions.shape[1]):
        for target in range(positions.shape[1]):
            if source == target:
                continue
            relation_rows.append(
                torch.stack(
                    (
                        cols[:, source] < cols[:, target],
                        cols[:, source] > cols[:, target],
                        rows[:, source] < rows[:, target],
                        rows[:, source] > rows[:, target],
                    ),
                    dim=-1,
                )
            )
    return torch.cat(relation_rows, dim=-1)


def evaluate_r440_room_layout(
    model: ConstructiveCognitiveCore,
    *,
    seed: int = 101,
    batch_size: int = 64,
) -> R440RoomLayoutResult:
    task = R440RoomLayout(seed=seed, batch_size=batch_size, config=model.config)
    frame1, frame2, extra, target = task.make_room_queries()
    with torch.no_grad():
        output = model(frame1, frame2, extra)
        prediction = output.answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        target_positions = target.view(task.config.colors, batch_size).T

        blank1 = torch.zeros_like(frame1)
        blank2 = torch.zeros_like(frame2)
        coord = task.curriculum.coord.unsqueeze(0).expand(frame1.shape[0], task.config.cells, 2)
        blank1[:, :, task.config.colors :] = coord
        blank2[:, :, task.config.colors :] = coord
        no_image_prediction = model(blank1, blank2, extra).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        slot_prediction = model(frame1, frame2, extra, ablate_slots=True).answer.argmax(dim=-1).view(task.config.colors, batch_size).T

    position_accuracy = float((prediction == target_positions).float().mean().item())
    exact_layout = float((prediction == target_positions).all(dim=-1).float().mean().item())
    no_overlap = float(
        torch.tensor([torch.unique(row).numel() == task.config.colors for row in prediction], dtype=torch.float32).mean().item()
    )
    interior = task.interior.to(prediction.device)
    avoids_wall = float(torch.isin(prediction, interior).float().mean().item())
    no_image_exact = float((no_image_prediction == target_positions).all(dim=-1).float().mean().item())
    slot_exact = float((slot_prediction == target_positions).all(dim=-1).float().mean().item())
    metrics = {
        "position_accuracy": position_accuracy,
        "exact_layout": exact_layout,
        "no_overlap": no_overlap,
        "avoids_wall": avoids_wall,
        "no_image_exact_layout": no_image_exact,
        "slot_ablation_exact_layout": slot_exact,
    }
    examples = []
    for idx in range(min(3, batch_size)):
        examples.append(
            {
                "expected": "/".join(task.render(target_positions[idx])),
                "predicted": "/".join(task.render(prediction[idx])),
            }
        )
    return R440RoomLayoutResult(
        seed=seed,
        passed=room_layout_passed(metrics),
        metrics=metrics,
        examples=examples,
    )


def evaluate_r440_room_update(
    model: ConstructiveCognitiveCore,
    *,
    seed: int = 101,
    batch_size: int = 64,
) -> R440RoomUpdateResult:
    task = R440RoomLayout(seed=seed, batch_size=batch_size, config=model.config)
    frame1, updated_frame2, stale_frame2, extra, target, moved_color = task.make_update_queries()
    with torch.no_grad():
        updated_output = model(frame1, updated_frame2, extra)
        updated_prediction = updated_output.answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        stale_prediction = model(frame1, stale_frame2, extra).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        slot_prediction = (
            model(frame1, updated_frame2, extra, ablate_slots=True).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        )
    target_positions = target.view(task.config.colors, batch_size).T
    updated_exact = (updated_prediction == target_positions).all(dim=-1)
    stale_exact = (stale_prediction == target_positions).all(dim=-1)
    slot_exact = (slot_prediction == target_positions).all(dim=-1)
    changed = updated_prediction[torch.arange(batch_size), moved_color] == target_positions[torch.arange(batch_size), moved_color]
    unchanged_scores = []
    for row in range(batch_size):
        mask = torch.ones(task.config.colors, dtype=torch.bool)
        mask[moved_color[row]] = False
        unchanged_scores.append(bool((updated_prediction[row, mask] == target_positions[row, mask]).all()))
    no_overlap = torch.tensor(
        [torch.unique(row).numel() == task.config.colors for row in updated_prediction],
        dtype=torch.float32,
    )
    interior = task.interior.to(updated_prediction.device)
    metrics = {
        "updated_exact_layout": float(updated_exact.float().mean().item()),
        "intended_object_changed": float(changed.float().mean().item()),
        "unchanged_objects_preserved": float(torch.tensor(unchanged_scores, dtype=torch.float32).mean().item()),
        "updated_no_overlap": float(no_overlap.mean().item()),
        "updated_avoids_wall": float(torch.isin(updated_prediction, interior).float().mean().item()),
        "stale_exact_layout": float(stale_exact.float().mean().item()),
        "slot_ablation_exact_layout": float(slot_exact.float().mean().item()),
    }
    examples = []
    for idx in range(min(3, batch_size)):
        examples.append(
            {
                "expected_updated": "/".join(task.render(target_positions[idx])),
                "predicted_updated": "/".join(task.render(updated_prediction[idx])),
                "stale_prediction": "/".join(task.render(stale_prediction[idx])),
            }
        )
    return R440RoomUpdateResult(
        seed=seed,
        passed=room_update_passed(metrics),
        metrics=metrics,
        examples=examples,
    )


def evaluate_r440_room_relations(
    model: ConstructiveCognitiveCore,
    *,
    seed: int = 101,
    batch_size: int = 64,
) -> R440RoomRelationResult:
    task = R440RoomLayout(seed=seed, batch_size=batch_size, config=model.config)
    frame1, frame2, extra, target = task.make_room_queries()
    with torch.no_grad():
        prediction = model(frame1, frame2, extra).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        no_image_frame1 = torch.zeros_like(frame1)
        no_image_frame2 = torch.zeros_like(frame2)
        coord = task.curriculum.coord.unsqueeze(0).expand(frame1.shape[0], task.config.cells, 2)
        no_image_frame1[:, :, task.config.colors :] = coord
        no_image_frame2[:, :, task.config.colors :] = coord
        no_image_prediction = (
            model(no_image_frame1, no_image_frame2, extra).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        )
        slot_prediction = model(frame1, frame2, extra, ablate_slots=True).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
    target_positions = target.view(task.config.colors, batch_size).T
    expected_relations = room_relation_tensor(target_positions, grid=task.config.grid)
    predicted_relations = room_relation_tensor(prediction, grid=task.config.grid)
    no_image_relations = room_relation_tensor(no_image_prediction, grid=task.config.grid)
    slot_relations = room_relation_tensor(slot_prediction, grid=task.config.grid)
    relation_accuracy = float((predicted_relations == expected_relations).float().mean().item())
    exact_relation_set = float((predicted_relations == expected_relations).all(dim=-1).float().mean().item())
    no_image_exact = float((no_image_relations == expected_relations).all(dim=-1).float().mean().item())
    slot_exact = float((slot_relations == expected_relations).all(dim=-1).float().mean().item())
    metrics = {
        "relation_accuracy": relation_accuracy,
        "exact_relation_set": exact_relation_set,
        "no_image_exact_relation_set": no_image_exact,
        "slot_ablation_exact_relation_set": slot_exact,
    }
    examples = []
    for idx in range(min(3, batch_size)):
        examples.append(
            {
                "layout": "/".join(task.render(target_positions[idx])),
                "predicted_layout": "/".join(task.render(prediction[idx])),
                "relations_match": str(bool((predicted_relations[idx] == expected_relations[idx]).all())),
            }
        )
    return R440RoomRelationResult(
        seed=seed,
        passed=room_relation_passed(metrics),
        metrics=metrics,
        examples=examples,
    )


def evaluate_r440_room_sequential_updates(
    model: ConstructiveCognitiveCore,
    *,
    seed: int = 101,
    batch_size: int = 64,
) -> R440RoomSequentialUpdateResult:
    task = R440RoomLayout(seed=seed, batch_size=batch_size, config=model.config)
    frame1, first_frame2, second_frame2, extra, target, first_color, second_color = task.make_sequential_update_queries()
    with torch.no_grad():
        final_prediction = model(frame1, second_frame2, extra).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        stale_prediction = model(frame1, first_frame2, extra).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        slot_prediction = (
            model(frame1, second_frame2, extra, ablate_slots=True).answer.argmax(dim=-1).view(task.config.colors, batch_size).T
        )
    target_positions = target.view(task.config.colors, batch_size).T
    final_exact = (final_prediction == target_positions).all(dim=-1)
    stale_exact = (stale_prediction == target_positions).all(dim=-1)
    slot_exact = (slot_prediction == target_positions).all(dim=-1)
    first_ok = final_prediction[torch.arange(batch_size), first_color] == target_positions[torch.arange(batch_size), first_color]
    second_ok = final_prediction[torch.arange(batch_size), second_color] == target_positions[torch.arange(batch_size), second_color]
    unmoved_scores = []
    for row in range(batch_size):
        mask = torch.ones(task.config.colors, dtype=torch.bool)
        mask[first_color[row]] = False
        mask[second_color[row]] = False
        unmoved_scores.append(bool((final_prediction[row, mask] == target_positions[row, mask]).all()))
    no_overlap = torch.tensor(
        [torch.unique(row).numel() == task.config.colors for row in final_prediction],
        dtype=torch.float32,
    )
    interior = task.interior.to(final_prediction.device)
    metrics = {
        "final_exact_layout": float(final_exact.float().mean().item()),
        "first_changed_object_correct": float(first_ok.float().mean().item()),
        "second_changed_object_correct": float(second_ok.float().mean().item()),
        "unmoved_object_preserved": float(torch.tensor(unmoved_scores, dtype=torch.float32).mean().item()),
        "final_no_overlap": float(no_overlap.mean().item()),
        "final_avoids_wall": float(torch.isin(final_prediction, interior).float().mean().item()),
        "intermediate_stale_exact_layout": float(stale_exact.float().mean().item()),
        "slot_ablation_exact_layout": float(slot_exact.float().mean().item()),
    }
    examples = []
    for idx in range(min(3, batch_size)):
        examples.append(
            {
                "expected_final": "/".join(task.render(target_positions[idx])),
                "predicted_final": "/".join(task.render(final_prediction[idx])),
                "intermediate_stale": "/".join(task.render(stale_prediction[idx])),
            }
        )
    return R440RoomSequentialUpdateResult(
        seed=seed,
        passed=room_sequential_update_passed(metrics),
        metrics=metrics,
        examples=examples,
    )
