"""Appearance-varying object files for R440 sensory states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .core import ConstructiveCoreConfig
from .room_layout import ROOM_DOOR, ROOM_EMPTY, ROOM_OBJECTS, ROOM_WALL


@dataclass(frozen=True)
class R440AppearanceObjectFileConfig:
    image_grid: int = 8
    image_channels: int = 8
    identity_vocab: int = 5
    shape_vocab: int = 3
    max_objects: int = 3
    hidden_dim: int = 224

    @property
    def image_dim(self) -> int:
        return self.image_grid * self.image_grid * self.image_channels

    @property
    def no_identity(self) -> int:
        return self.identity_vocab

    @property
    def no_shape(self) -> int:
        return self.shape_vocab

    @property
    def identity_classes(self) -> int:
        return self.identity_vocab + 1

    @property
    def shape_classes(self) -> int:
        return self.shape_vocab + 1


@dataclass(frozen=True)
class R440AppearanceObjectBatch:
    image: torch.Tensor
    identities: torch.Tensor
    shapes: torch.Tensor
    positions: torch.Tensor
    present: torch.Tensor


@dataclass
class R440AppearanceObjectOutput:
    identity_logits: torch.Tensor
    shape_logits: torch.Tensor
    position_logits: torch.Tensor
    presence_logits: torch.Tensor


@dataclass(frozen=True)
class R440AppearanceObjectResult:
    seed: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


class R440AppearanceObjectBridge(nn.Module):
    def __init__(
        self,
        *,
        core_config: ConstructiveCoreConfig | None = None,
        object_config: R440AppearanceObjectFileConfig | None = None,
    ):
        super().__init__()
        self.core_config = core_config or ConstructiveCoreConfig()
        self.object_config = object_config or R440AppearanceObjectFileConfig()
        cfg = self.object_config
        self.net = nn.Sequential(
            nn.Linear(cfg.image_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.identity_head = nn.Linear(cfg.hidden_dim, cfg.max_objects * cfg.identity_classes)
        self.shape_head = nn.Linear(cfg.hidden_dim, cfg.max_objects * cfg.shape_classes)
        self.position_head = nn.Linear(cfg.hidden_dim, cfg.max_objects * self.core_config.cells)
        self.presence_head = nn.Linear(cfg.hidden_dim, cfg.max_objects)

    def forward(self, image: torch.Tensor) -> R440AppearanceObjectOutput:
        cfg = self.object_config
        hidden = self.net(image.reshape(image.shape[0], -1))
        return R440AppearanceObjectOutput(
            identity_logits=self.identity_head(hidden).view(image.shape[0], cfg.max_objects, cfg.identity_classes),
            shape_logits=self.shape_head(hidden).view(image.shape[0], cfg.max_objects, cfg.shape_classes),
            position_logits=self.position_head(hidden).view(image.shape[0], cfg.max_objects, self.core_config.cells),
            presence_logits=self.presence_head(hidden),
        )


class R440AppearanceObjectRenderer:
    def __init__(
        self,
        *,
        seed: int,
        batch_size: int = 64,
        core_config: ConstructiveCoreConfig | None = None,
        object_config: R440AppearanceObjectFileConfig | None = None,
    ):
        torch.manual_seed(seed)
        self.seed = seed
        self.batch_size = batch_size
        self.core_config = core_config or ConstructiveCoreConfig()
        self.object_config = object_config or R440AppearanceObjectFileConfig()
        if self.core_config.grid != 4:
            raise ValueError("R440AppearanceObjectRenderer currently expects a 4x4 grid")
        if self.object_config.image_grid % self.core_config.grid != 0:
            raise ValueError("image grid must be an integer multiple of the room grid")
        self.scale = self.object_config.image_grid // self.core_config.grid
        self.interior = torch.tensor([5, 6, 9, 10], dtype=torch.long)
        self.wall = torch.tensor([0, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15], dtype=torch.long)
        self.door = 1

    def make_batch(self, *, delete_one: bool = False) -> R440AppearanceObjectBatch:
        cfg = self.object_config
        identity_rows: list[torch.Tensor] = []
        shape_rows: list[torch.Tensor] = []
        position_rows: list[torch.Tensor] = []
        present_rows: list[torch.Tensor] = []
        for _ in range(self.batch_size):
            count = int(torch.randint(1, cfg.max_objects + 1, (1,)).item())
            identities = torch.randperm(cfg.identity_vocab)[:count].sort().values
            shapes = torch.randint(0, cfg.shape_vocab, (count,))
            positions = self.interior[torch.randperm(self.interior.numel())[:count]]
            if delete_one and count > 0:
                keep = torch.ones(count, dtype=torch.bool)
                keep[int(torch.randint(0, count, (1,)).item())] = False
                identities = identities[keep]
                shapes = shapes[keep]
                positions = positions[keep]
            ids, shp, pos, present = self._pack(identities, shapes, positions)
            identity_rows.append(ids)
            shape_rows.append(shp)
            position_rows.append(pos)
            present_rows.append(present)
        identities = torch.stack(identity_rows)
        shapes = torch.stack(shape_rows)
        positions = torch.stack(position_rows)
        present = torch.stack(present_rows)
        return R440AppearanceObjectBatch(
            image=self.render(identities, shapes, positions, present),
            identities=identities,
            shapes=shapes,
            positions=positions,
            present=present,
        )

    def blank_batch(self) -> R440AppearanceObjectBatch:
        cfg = self.object_config
        identities = torch.full((self.batch_size, cfg.max_objects), cfg.no_identity, dtype=torch.long)
        shapes = torch.full((self.batch_size, cfg.max_objects), cfg.no_shape, dtype=torch.long)
        positions = torch.zeros(self.batch_size, cfg.max_objects, dtype=torch.long)
        present = torch.zeros(self.batch_size, cfg.max_objects, dtype=torch.bool)
        return R440AppearanceObjectBatch(
            image=torch.zeros(self.batch_size, cfg.image_channels, cfg.image_grid, cfg.image_grid),
            identities=identities,
            shapes=shapes,
            positions=positions,
            present=present,
        )

    def translate(self, batch: R440AppearanceObjectBatch) -> R440AppearanceObjectBatch:
        positions = batch.positions.clone()
        for row in range(positions.shape[0]):
            visible = torch.where(batch.present[row])[0]
            if visible.numel() == 0:
                continue
            occupied = set(int(cell) for cell in positions[row, visible].tolist())
            empty = [int(cell) for cell in self.interior.tolist() if int(cell) not in occupied]
            positions[row, int(visible[0])] = empty[0]
        return R440AppearanceObjectBatch(
            image=self.render(batch.identities, batch.shapes, positions, batch.present),
            identities=batch.identities,
            shapes=batch.shapes,
            positions=positions,
            present=batch.present,
        )

    def render(
        self,
        identities: torch.Tensor,
        shapes: torch.Tensor,
        positions: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.object_config
        image = torch.zeros(identities.shape[0], cfg.image_channels, cfg.image_grid, cfg.image_grid)
        for cell in self.wall.tolist():
            self._paint(image, 0, cell // self.core_config.grid, cell % self.core_config.grid)
        self._paint(image, 1, self.door // self.core_config.grid, self.door % self.core_config.grid)
        for row in range(identities.shape[0]):
            for slot in range(cfg.max_objects):
                if not bool(present[row, slot]):
                    continue
                cell = int(positions[row, slot])
                room_row = cell // self.core_config.grid
                room_col = cell % self.core_config.grid
                self._paint(image[row : row + 1], 2, room_row, room_col)
                identity_code = int(identities[row, slot]) + 1
                for bit in range(3):
                    if identity_code & (1 << bit):
                        self._paint(image[row : row + 1], 3 + bit, room_row, room_col)
                shape_code = int(shapes[row, slot]) + 1
                for bit in range(2):
                    if shape_code & (1 << bit):
                        self._paint(image[row : row + 1], 6 + bit, room_row, room_col)
        return image

    def render_wireframe(
        self,
        identities: torch.Tensor,
        shapes: torch.Tensor,
        positions: torch.Tensor,
        present: torch.Tensor,
    ) -> list[str]:
        cells = [ROOM_EMPTY for _ in range(self.core_config.cells)]
        for idx in self.wall.tolist():
            cells[idx] = ROOM_WALL
        cells[self.door] = ROOM_DOOR
        symbols = ROOM_OBJECTS + ("Y", "M")
        shape_marks = ("o", "s", "t")
        for slot in range(self.object_config.max_objects):
            if not bool(present[slot]):
                continue
            identity = int(identities[slot])
            shape = int(shapes[slot])
            cell = int(positions[slot])
            if 0 <= identity < len(symbols) and 0 <= shape < len(shape_marks) and 0 <= cell < self.core_config.cells:
                cells[cell] = symbols[identity].lower() if shape == 0 else symbols[identity]
        return ["".join(cells[row * self.core_config.grid : (row + 1) * self.core_config.grid]) for row in range(self.core_config.grid)]

    def file_string(self, identities: torch.Tensor, shapes: torch.Tensor, positions: torch.Tensor, present: torch.Tensor) -> str:
        parts = []
        for slot in range(self.object_config.max_objects):
            if bool(present[slot]):
                parts.append(f"id{int(identities[slot])}:shape{int(shapes[slot])}@{int(positions[slot])}")
        return ",".join(parts) if parts else "empty"

    def _paint(self, image: torch.Tensor, channel: int, row: int, col: int) -> None:
        r0 = row * self.scale
        c0 = col * self.scale
        image[:, channel, r0 : r0 + self.scale, c0 : c0 + self.scale] = 1.0

    def _pack(
        self,
        identities: torch.Tensor,
        shapes: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.object_config
        ids = torch.full((cfg.max_objects,), cfg.no_identity, dtype=torch.long)
        shp = torch.full((cfg.max_objects,), cfg.no_shape, dtype=torch.long)
        pos = torch.zeros(cfg.max_objects, dtype=torch.long)
        present = torch.zeros(cfg.max_objects, dtype=torch.bool)
        if identities.numel() == 0:
            return ids, shp, pos, present
        order = torch.argsort(identities)
        identities = identities[order]
        shapes = shapes[order]
        positions = positions[order]
        count = min(int(identities.numel()), cfg.max_objects)
        ids[:count] = identities[:count]
        shp[:count] = shapes[:count]
        pos[:count] = positions[:count]
        present[:count] = True
        return ids, shp, pos, present


def r440_appearance_object_loss(output: R440AppearanceObjectOutput, batch: R440AppearanceObjectBatch) -> torch.Tensor:
    visible = batch.present.reshape(-1)
    flat_positions = batch.positions.reshape(-1)
    flat_position_logits = output.position_logits.reshape(-1, output.position_logits.shape[-1])
    if bool(visible.any()):
        position_loss = F.cross_entropy(flat_position_logits[visible], flat_positions[visible])
    else:
        position_loss = output.position_logits.sum() * 0.0
    identity_loss = F.cross_entropy(
        output.identity_logits.reshape(-1, output.identity_logits.shape[-1]),
        batch.identities.reshape(-1),
    )
    shape_loss = F.cross_entropy(
        output.shape_logits.reshape(-1, output.shape_logits.shape[-1]),
        batch.shapes.reshape(-1),
    )
    presence_loss = F.binary_cross_entropy_with_logits(output.presence_logits, batch.present.float())
    return position_loss + identity_loss + shape_loss + presence_loss


def appearance_object_predictions(output: R440AppearanceObjectOutput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        output.identity_logits.argmax(dim=-1),
        output.shape_logits.argmax(dim=-1),
        output.position_logits.argmax(dim=-1),
        output.presence_logits > 0,
    )


def appearance_object_exact(
    predicted_identities: torch.Tensor,
    predicted_shapes: torch.Tensor,
    predicted_positions: torch.Tensor,
    predicted_present: torch.Tensor,
    target: R440AppearanceObjectBatch,
) -> torch.Tensor:
    presence_match = predicted_present == target.present
    identity_match = predicted_identities == target.identities
    shape_match = predicted_shapes == target.shapes
    position_match = torch.where(target.present, predicted_positions == target.positions, torch.ones_like(target.present))
    return (presence_match & identity_match & shape_match & position_match).all(dim=-1)


def appearance_relation_tensor(
    identities: torch.Tensor,
    positions: torch.Tensor,
    present: torch.Tensor,
    *,
    identity_vocab: int,
    grid: int = 4,
) -> torch.Tensor:
    rows = positions // grid
    cols = positions % grid
    relation_rows = []
    for source in range(identity_vocab):
        source_mask = (identities == source) & present
        source_row = torch.where(source_mask, rows, torch.zeros_like(rows)).sum(dim=-1)
        source_col = torch.where(source_mask, cols, torch.zeros_like(cols)).sum(dim=-1)
        source_present = source_mask.any(dim=-1)
        for target in range(identity_vocab):
            if source == target:
                continue
            target_mask = (identities == target) & present
            target_row = torch.where(target_mask, rows, torch.zeros_like(rows)).sum(dim=-1)
            target_col = torch.where(target_mask, cols, torch.zeros_like(cols)).sum(dim=-1)
            target_present = target_mask.any(dim=-1)
            pair_present = source_present & target_present
            relation_rows.append(
                torch.stack(
                    (
                        pair_present & (source_col < target_col),
                        pair_present & (source_col > target_col),
                        pair_present & (source_row < target_row),
                        pair_present & (source_row > target_row),
                    ),
                    dim=-1,
                )
            )
    return torch.cat(relation_rows, dim=-1)


def evaluate_r440_appearance_object_files(
    bridge: R440AppearanceObjectBridge,
    renderer: R440AppearanceObjectRenderer,
) -> R440AppearanceObjectResult:
    batch = renderer.make_batch()
    deleted = renderer.make_batch(delete_one=True)
    translated = renderer.translate(batch)
    blank = renderer.blank_batch()
    with torch.no_grad():
        pred_ids, pred_shapes, pred_pos, pred_present = appearance_object_predictions(bridge(batch.image))
        delete_ids, delete_shapes, delete_pos, delete_present = appearance_object_predictions(bridge(deleted.image))
        translated_ids, translated_shapes, translated_pos, translated_present = appearance_object_predictions(bridge(translated.image))
        blank_ids, blank_shapes, blank_pos, blank_present = appearance_object_predictions(bridge(blank.image))
        identity_ablated = batch.image.clone()
        identity_ablated[:, 3:6, :, :] = 0.0
        identity_ablated_ids, identity_ablated_shapes, identity_ablated_pos, identity_ablated_present = appearance_object_predictions(
            bridge(identity_ablated)
        )
        shape_ablated = batch.image.clone()
        shape_ablated[:, 6:8, :, :] = 0.0
        shape_ablated_ids, shape_ablated_shapes, shape_ablated_pos, shape_ablated_present = appearance_object_predictions(
            bridge(shape_ablated)
        )

    visible_count = batch.present.float().sum().clamp_min(1.0)
    identity_accuracy = ((pred_ids == batch.identities) & batch.present).float().sum() / visible_count
    shape_accuracy = ((pred_shapes == batch.shapes) & batch.present).float().sum() / visible_count
    position_accuracy = ((pred_pos == batch.positions) & batch.present).float().sum() / visible_count
    exact_files = appearance_object_exact(pred_ids, pred_shapes, pred_pos, pred_present, batch)
    delete_exact = appearance_object_exact(delete_ids, delete_shapes, delete_pos, delete_present, deleted)
    translated_exact = appearance_object_exact(translated_ids, translated_shapes, translated_pos, translated_present, translated)
    identity_ablated_exact = appearance_object_exact(
        identity_ablated_ids,
        identity_ablated_shapes,
        identity_ablated_pos,
        identity_ablated_present,
        batch,
    )
    shape_ablated_exact = appearance_object_exact(
        shape_ablated_ids,
        shape_ablated_shapes,
        shape_ablated_pos,
        shape_ablated_present,
        batch,
    )
    expected_relations = appearance_relation_tensor(
        batch.identities,
        batch.positions,
        batch.present,
        identity_vocab=renderer.object_config.identity_vocab,
        grid=renderer.core_config.grid,
    )
    predicted_relations = appearance_relation_tensor(
        pred_ids,
        pred_pos,
        pred_present,
        identity_vocab=renderer.object_config.identity_vocab,
        grid=renderer.core_config.grid,
    )
    metrics = {
        "identity_accuracy": float(identity_accuracy.item()),
        "shape_accuracy": float(shape_accuracy.item()),
        "position_accuracy": float(position_accuracy.item()),
        "presence_accuracy": float((pred_present == batch.present).float().mean().item()),
        "exact_object_files": float(exact_files.float().mean().item()),
        "relation_accuracy": float((predicted_relations == expected_relations).float().mean().item()),
        "translated_exact_object_files": float(translated_exact.float().mean().item()),
        "translated_changed_position": float((translated.positions != batch.positions).any(dim=-1).float().mean().item()),
        "delete_exact_object_files": float(delete_exact.float().mean().item()),
        "delete_presence_accuracy": float((delete_present == deleted.present).float().mean().item()),
        "blank_presence_rate": float(blank_present.float().mean().item()),
        "identity_ablation_exact_object_files": float(identity_ablated_exact.float().mean().item()),
        "identity_ablation_identity_accuracy": float(
            (((identity_ablated_ids == batch.identities) & batch.present).float().sum() / visible_count).item()
        ),
        "shape_ablation_exact_object_files": float(shape_ablated_exact.float().mean().item()),
        "shape_ablation_shape_accuracy": float(
            (((shape_ablated_shapes == batch.shapes) & batch.present).float().sum() / visible_count).item()
        ),
    }
    examples = []
    for idx in range(min(3, renderer.batch_size)):
        examples.append(
            {
                "expected": renderer.file_string(batch.identities[idx], batch.shapes[idx], batch.positions[idx], batch.present[idx]),
                "predicted": renderer.file_string(pred_ids[idx], pred_shapes[idx], pred_pos[idx], pred_present[idx]),
                "translated_expected": renderer.file_string(
                    translated.identities[idx],
                    translated.shapes[idx],
                    translated.positions[idx],
                    translated.present[idx],
                ),
                "translated_predicted": renderer.file_string(
                    translated_ids[idx],
                    translated_shapes[idx],
                    translated_pos[idx],
                    translated_present[idx],
                ),
            }
        )
    return R440AppearanceObjectResult(
        seed=renderer.seed,
        passed=r440_appearance_object_files_passed(metrics),
        metrics=metrics,
        examples=examples,
    )


def r440_appearance_object_files_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["identity_accuracy"] >= 0.95
        and metrics["shape_accuracy"] >= 0.95
        and metrics["position_accuracy"] >= 0.95
        and metrics["presence_accuracy"] >= 0.95
        and metrics["exact_object_files"] >= 0.90
        and metrics["relation_accuracy"] >= 0.95
        and metrics["translated_exact_object_files"] >= 0.90
        and metrics["translated_changed_position"] >= 0.90
        and metrics["delete_exact_object_files"] >= 0.90
        and metrics["delete_presence_accuracy"] >= 0.90
        and metrics["blank_presence_rate"] <= 0.20
        and metrics["identity_ablation_exact_object_files"] <= 0.35
        and metrics["identity_ablation_identity_accuracy"] <= 0.60
        and metrics["shape_ablation_exact_object_files"] <= 0.35
        and metrics["shape_ablation_shape_accuracy"] <= 0.60
    )


def train_r440_appearance_object_files(
    *,
    seed: int = 101,
    batch_size: int = 64,
    steps: int = 2400,
    lr: float = 0.004,
) -> tuple[R440AppearanceObjectBridge, R440AppearanceObjectRenderer, R440AppearanceObjectResult]:
    torch.manual_seed(seed)
    bridge = R440AppearanceObjectBridge()
    renderer = R440AppearanceObjectRenderer(seed=seed, batch_size=batch_size, core_config=bridge.core_config)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr)
    bridge.train()
    for step in range(steps):
        if step % 7 == 0:
            batch = renderer.blank_batch()
        else:
            batch = renderer.make_batch(delete_one=step % 5 == 0)
        loss = r440_appearance_object_loss(bridge(batch.image), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    bridge.eval()
    result = evaluate_r440_appearance_object_files(
        bridge,
        R440AppearanceObjectRenderer(seed=seed + 11000, batch_size=batch_size, core_config=bridge.core_config),
    )
    return bridge, renderer, result
