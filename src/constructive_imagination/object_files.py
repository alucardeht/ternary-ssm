"""Variable-identity object-file bridge for R440 sensory states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .core import ConstructiveCoreConfig
from .room_layout import ROOM_DOOR, ROOM_EMPTY, ROOM_OBJECTS, ROOM_WALL


@dataclass(frozen=True)
class R440VariableObjectFileConfig:
    image_grid: int = 8
    image_channels: int = 6
    identity_vocab: int = 5
    max_objects: int = 3
    hidden_dim: int = 192

    @property
    def image_dim(self) -> int:
        return self.image_grid * self.image_grid * self.image_channels

    @property
    def no_identity(self) -> int:
        return self.identity_vocab

    @property
    def identity_classes(self) -> int:
        return self.identity_vocab + 1


@dataclass(frozen=True)
class R440VariableObjectBatch:
    image: torch.Tensor
    identities: torch.Tensor
    positions: torch.Tensor
    present: torch.Tensor


@dataclass
class R440VariableObjectFileOutput:
    identity_logits: torch.Tensor
    position_logits: torch.Tensor
    presence_logits: torch.Tensor


@dataclass(frozen=True)
class R440VariableObjectFileResult:
    seed: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


class R440VariableObjectFileBridge(nn.Module):
    def __init__(
        self,
        *,
        core_config: ConstructiveCoreConfig | None = None,
        object_config: R440VariableObjectFileConfig | None = None,
    ):
        super().__init__()
        self.core_config = core_config or ConstructiveCoreConfig()
        self.object_config = object_config or R440VariableObjectFileConfig()
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
        self.position_head = nn.Linear(cfg.hidden_dim, cfg.max_objects * self.core_config.cells)
        self.presence_head = nn.Linear(cfg.hidden_dim, cfg.max_objects)

    def forward(self, image: torch.Tensor) -> R440VariableObjectFileOutput:
        cfg = self.object_config
        hidden = self.net(image.reshape(image.shape[0], -1))
        return R440VariableObjectFileOutput(
            identity_logits=self.identity_head(hidden).view(image.shape[0], cfg.max_objects, cfg.identity_classes),
            position_logits=self.position_head(hidden).view(image.shape[0], cfg.max_objects, self.core_config.cells),
            presence_logits=self.presence_head(hidden),
        )


class R440VariableObjectRenderer:
    def __init__(
        self,
        *,
        seed: int,
        batch_size: int = 64,
        core_config: ConstructiveCoreConfig | None = None,
        object_config: R440VariableObjectFileConfig | None = None,
    ):
        torch.manual_seed(seed)
        self.seed = seed
        self.batch_size = batch_size
        self.core_config = core_config or ConstructiveCoreConfig()
        self.object_config = object_config or R440VariableObjectFileConfig()
        if self.core_config.grid != 4:
            raise ValueError("R440VariableObjectRenderer currently expects a 4x4 grid")
        if self.object_config.image_grid % self.core_config.grid != 0:
            raise ValueError("image grid must be an integer multiple of the room grid")
        self.scale = self.object_config.image_grid // self.core_config.grid
        self.interior = torch.tensor([5, 6, 9, 10], dtype=torch.long)
        self.wall = torch.tensor([0, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15], dtype=torch.long)
        self.door = 1

    def make_batch(self, *, delete_one: bool = False) -> R440VariableObjectBatch:
        identity_rows: list[torch.Tensor] = []
        position_rows: list[torch.Tensor] = []
        present_rows: list[torch.Tensor] = []
        cfg = self.object_config
        for _ in range(self.batch_size):
            count = int(torch.randint(1, cfg.max_objects + 1, (1,)).item())
            identities = torch.randperm(cfg.identity_vocab)[:count].sort().values
            positions = self.interior[torch.randperm(self.interior.numel())[:count]]
            if delete_one and count > 0:
                keep = torch.ones(count, dtype=torch.bool)
                keep[int(torch.randint(0, count, (1,)).item())] = False
                identities = identities[keep]
                positions = positions[keep]
            ids, pos, present = self._pack(identities, positions)
            identity_rows.append(ids)
            position_rows.append(pos)
            present_rows.append(present)
        identities = torch.stack(identity_rows)
        positions = torch.stack(position_rows)
        present = torch.stack(present_rows)
        return R440VariableObjectBatch(
            image=self.render(identities, positions, present),
            identities=identities,
            positions=positions,
            present=present,
        )

    def blank_batch(self) -> R440VariableObjectBatch:
        cfg = self.object_config
        identities = torch.full((self.batch_size, cfg.max_objects), cfg.no_identity, dtype=torch.long)
        positions = torch.zeros(self.batch_size, cfg.max_objects, dtype=torch.long)
        present = torch.zeros(self.batch_size, cfg.max_objects, dtype=torch.bool)
        return R440VariableObjectBatch(
            image=torch.zeros(self.batch_size, cfg.image_channels, cfg.image_grid, cfg.image_grid),
            identities=identities,
            positions=positions,
            present=present,
        )

    def translate(self, batch: R440VariableObjectBatch) -> R440VariableObjectBatch:
        positions = batch.positions.clone()
        for row in range(positions.shape[0]):
            visible = torch.where(batch.present[row])[0]
            if visible.numel() == 0:
                continue
            occupied = set(int(cell) for cell in positions[row, visible].tolist())
            empty = [int(cell) for cell in self.interior.tolist() if int(cell) not in occupied]
            positions[row, int(visible[0])] = empty[0]
        return R440VariableObjectBatch(
            image=self.render(batch.identities, positions, batch.present),
            identities=batch.identities,
            positions=positions,
            present=batch.present,
        )

    def render(self, identities: torch.Tensor, positions: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
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
        return image

    def render_wireframe(self, identities: torch.Tensor, positions: torch.Tensor, present: torch.Tensor) -> list[str]:
        cells = [ROOM_EMPTY for _ in range(self.core_config.cells)]
        for idx in self.wall.tolist():
            cells[idx] = ROOM_WALL
        cells[self.door] = ROOM_DOOR
        symbols = ROOM_OBJECTS + ("Y", "M")
        for slot in range(self.object_config.max_objects):
            if not bool(present[slot]):
                continue
            identity = int(identities[slot])
            cell = int(positions[slot])
            if 0 <= identity < len(symbols) and 0 <= cell < self.core_config.cells:
                cells[cell] = symbols[identity]
        return ["".join(cells[row * self.core_config.grid : (row + 1) * self.core_config.grid]) for row in range(self.core_config.grid)]

    def _paint(self, image: torch.Tensor, channel: int, row: int, col: int) -> None:
        r0 = row * self.scale
        c0 = col * self.scale
        image[:, channel, r0 : r0 + self.scale, c0 : c0 + self.scale] = 1.0

    def _pack(self, identities: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.object_config
        ids = torch.full((cfg.max_objects,), cfg.no_identity, dtype=torch.long)
        pos = torch.zeros(cfg.max_objects, dtype=torch.long)
        present = torch.zeros(cfg.max_objects, dtype=torch.bool)
        if identities.numel() == 0:
            return ids, pos, present
        order = torch.argsort(identities)
        identities = identities[order]
        positions = positions[order]
        count = min(int(identities.numel()), cfg.max_objects)
        ids[:count] = identities[:count]
        pos[:count] = positions[:count]
        present[:count] = True
        return ids, pos, present


def r440_variable_object_file_loss(
    output: R440VariableObjectFileOutput,
    batch: R440VariableObjectBatch,
) -> torch.Tensor:
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
    presence_loss = F.binary_cross_entropy_with_logits(output.presence_logits, batch.present.float())
    return position_loss + identity_loss + presence_loss


def variable_object_predictions(output: R440VariableObjectFileOutput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        output.identity_logits.argmax(dim=-1),
        output.position_logits.argmax(dim=-1),
        output.presence_logits > 0,
    )


def object_file_exact(
    predicted_identities: torch.Tensor,
    predicted_positions: torch.Tensor,
    predicted_present: torch.Tensor,
    target: R440VariableObjectBatch,
) -> torch.Tensor:
    presence_match = predicted_present == target.present
    identity_match = torch.where(target.present, predicted_identities == target.identities, predicted_identities == target.identities)
    position_match = torch.where(target.present, predicted_positions == target.positions, torch.ones_like(target.present))
    return (presence_match & identity_match & position_match).all(dim=-1)


def variable_relation_tensor(
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


def evaluate_r440_variable_object_files(
    bridge: R440VariableObjectFileBridge,
    renderer: R440VariableObjectRenderer,
) -> R440VariableObjectFileResult:
    batch = renderer.make_batch()
    deleted = renderer.make_batch(delete_one=True)
    translated = renderer.translate(batch)
    blank = renderer.blank_batch()
    with torch.no_grad():
        pred_ids, pred_pos, pred_present = variable_object_predictions(bridge(batch.image))
        delete_ids, delete_pos, delete_present = variable_object_predictions(bridge(deleted.image))
        translated_ids, translated_pos, translated_present = variable_object_predictions(bridge(translated.image))
        blank_ids, blank_pos, blank_present = variable_object_predictions(bridge(blank.image))
        identity_ablated = batch.image.clone()
        identity_ablated[:, 3:, :, :] = 0.0
        ablated_ids, ablated_pos, ablated_present = variable_object_predictions(bridge(identity_ablated))

    present_count = batch.present.float().sum().clamp_min(1.0)
    visible_identity = ((pred_ids == batch.identities) & batch.present).float().sum() / present_count
    visible_position = ((pred_pos == batch.positions) & batch.present).float().sum() / present_count
    exact_files = object_file_exact(pred_ids, pred_pos, pred_present, batch)
    translated_exact = object_file_exact(translated_ids, translated_pos, translated_present, translated)
    delete_exact = object_file_exact(delete_ids, delete_pos, delete_present, deleted)
    ablated_exact = object_file_exact(ablated_ids, ablated_pos, ablated_present, batch)
    relation_expected = variable_relation_tensor(
        batch.identities,
        batch.positions,
        batch.present,
        identity_vocab=renderer.object_config.identity_vocab,
        grid=renderer.core_config.grid,
    )
    relation_predicted = variable_relation_tensor(
        pred_ids,
        pred_pos,
        pred_present,
        identity_vocab=renderer.object_config.identity_vocab,
        grid=renderer.core_config.grid,
    )
    ablated_identity_accuracy = float((((ablated_ids == batch.identities) & batch.present).float().sum() / present_count).item())
    metrics = {
        "identity_accuracy": float(visible_identity.item()),
        "position_accuracy": float(visible_position.item()),
        "presence_accuracy": float((pred_present == batch.present).float().mean().item()),
        "exact_object_files": float(exact_files.float().mean().item()),
        "relation_accuracy": float((relation_predicted == relation_expected).float().mean().item()),
        "translated_exact_object_files": float(translated_exact.float().mean().item()),
        "translated_changed_position": float((translated.positions != batch.positions).any(dim=-1).float().mean().item()),
        "delete_exact_object_files": float(delete_exact.float().mean().item()),
        "delete_presence_accuracy": float((delete_present == deleted.present).float().mean().item()),
        "blank_presence_rate": float(blank_present.float().mean().item()),
        "identity_ablation_exact_object_files": float(ablated_exact.float().mean().item()),
        "identity_ablation_identity_accuracy": ablated_identity_accuracy,
    }
    examples = []
    for idx in range(min(3, renderer.batch_size)):
        examples.append(
            {
                "expected": "/".join(renderer.render_wireframe(batch.identities[idx], batch.positions[idx], batch.present[idx])),
                "predicted": "/".join(renderer.render_wireframe(pred_ids[idx], pred_pos[idx], pred_present[idx])),
                "translated_expected": "/".join(
                    renderer.render_wireframe(translated.identities[idx], translated.positions[idx], translated.present[idx])
                ),
                "translated_predicted": "/".join(
                    renderer.render_wireframe(translated_ids[idx], translated_pos[idx], translated_present[idx])
                ),
            }
        )
    return R440VariableObjectFileResult(
        seed=renderer.seed,
        passed=r440_variable_object_files_passed(metrics),
        metrics=metrics,
        examples=examples,
    )


def r440_variable_object_files_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["identity_accuracy"] >= 0.95
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
    )


def train_r440_variable_object_files(
    *,
    seed: int = 101,
    batch_size: int = 64,
    steps: int = 900,
    lr: float = 0.005,
) -> tuple[R440VariableObjectFileBridge, R440VariableObjectRenderer, R440VariableObjectFileResult]:
    torch.manual_seed(seed)
    bridge = R440VariableObjectFileBridge()
    renderer = R440VariableObjectRenderer(seed=seed, batch_size=batch_size, core_config=bridge.core_config)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr)
    bridge.train()
    for step in range(steps):
        if step % 7 == 0:
            batch = renderer.blank_batch()
        else:
            batch = renderer.make_batch(delete_one=step % 5 == 0)
        loss = r440_variable_object_file_loss(bridge(batch.image), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    bridge.eval()
    result = evaluate_r440_variable_object_files(
        bridge,
        R440VariableObjectRenderer(seed=seed + 10000, batch_size=batch_size, core_config=bridge.core_config),
    )
    return bridge, renderer, result
