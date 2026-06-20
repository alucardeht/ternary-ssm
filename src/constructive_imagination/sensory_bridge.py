"""Object-centric sensory bridge for R440 room states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .core import ConstructiveCoreConfig
from .room_layout import R440RoomLayout, room_relation_tensor


@dataclass(frozen=True)
class R440SensoryBridgeConfig:
    image_grid: int = 8
    image_channels: int = 5
    object_count: int = 3
    hidden_dim: int = 128

    @property
    def image_dim(self) -> int:
        return self.image_grid * self.image_grid * self.image_channels


@dataclass(frozen=True)
class R440SensoryBatch:
    image: torch.Tensor
    positions: torch.Tensor
    present: torch.Tensor


@dataclass
class R440SensoryBridgeOutput:
    position_logits: torch.Tensor
    presence_logits: torch.Tensor


@dataclass(frozen=True)
class R440SensoryBridgeResult:
    seed: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


class R440SensoryBridge(nn.Module):
    def __init__(
        self,
        *,
        core_config: ConstructiveCoreConfig | None = None,
        bridge_config: R440SensoryBridgeConfig | None = None,
    ):
        super().__init__()
        self.core_config = core_config or ConstructiveCoreConfig()
        self.bridge_config = bridge_config or R440SensoryBridgeConfig()
        self.net = nn.Sequential(
            nn.Linear(self.bridge_config.image_dim, self.bridge_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.bridge_config.hidden_dim, self.bridge_config.hidden_dim),
            nn.ReLU(),
        )
        self.position_head = nn.Linear(
            self.bridge_config.hidden_dim,
            self.bridge_config.object_count * self.core_config.cells,
        )
        self.presence_head = nn.Linear(self.bridge_config.hidden_dim, self.bridge_config.object_count)

    def forward(self, image: torch.Tensor) -> R440SensoryBridgeOutput:
        hidden = self.net(image.reshape(image.shape[0], -1))
        return R440SensoryBridgeOutput(
            position_logits=self.position_head(hidden).view(
                image.shape[0],
                self.bridge_config.object_count,
                self.core_config.cells,
            ),
            presence_logits=self.presence_head(hidden),
        )


class R440SensoryRoomRenderer:
    def __init__(
        self,
        *,
        seed: int,
        batch_size: int = 64,
        core_config: ConstructiveCoreConfig | None = None,
        bridge_config: R440SensoryBridgeConfig | None = None,
    ):
        self.room = R440RoomLayout(seed=seed, batch_size=batch_size, config=core_config or ConstructiveCoreConfig())
        self.bridge_config = bridge_config or R440SensoryBridgeConfig()
        self.batch_size = batch_size
        self.scale = self.bridge_config.image_grid // self.room.config.grid
        if self.bridge_config.image_grid % self.room.config.grid != 0:
            raise ValueError("image grid must be an integer multiple of the room grid")

    def render(self, positions: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        present = torch.ones_like(positions, dtype=torch.bool) if present is None else present.bool()
        image = torch.zeros(positions.shape[0], self.bridge_config.image_channels, self.bridge_config.image_grid, self.bridge_config.image_grid)
        wall_channel = 0
        door_channel = 1
        object_offset = 2
        for cell in self.room.wall.tolist():
            row = cell // self.room.config.grid
            col = cell % self.room.config.grid
            self._paint(image, wall_channel, row, col)
        row = self.room.door // self.room.config.grid
        col = self.room.door % self.room.config.grid
        self._paint(image, door_channel, row, col)
        for obj in range(self.room.config.colors):
            for batch in range(positions.shape[0]):
                if not bool(present[batch, obj]):
                    continue
                cell = int(positions[batch, obj])
                row = cell // self.room.config.grid
                col = cell % self.room.config.grid
                self._paint(image[batch : batch + 1], object_offset + obj, row, col)
        return image

    def _paint(self, image: torch.Tensor, channel: int, row: int, col: int) -> None:
        r0 = row * self.scale
        c0 = col * self.scale
        image[:, channel, r0 : r0 + self.scale, c0 : c0 + self.scale] = 1.0

    def make_batch(self, *, delete_one: bool = False) -> R440SensoryBatch:
        positions = self.room.sample_positions()
        present = torch.ones(self.batch_size, self.room.config.colors, dtype=torch.bool)
        if delete_one:
            removed = torch.randint(0, self.room.config.colors, (self.batch_size,))
            present[torch.arange(self.batch_size), removed] = False
        return R440SensoryBatch(image=self.render(positions, present), positions=positions, present=present)

    def translate(self, positions: torch.Tensor) -> torch.Tensor:
        shifted = positions.clone()
        for batch in range(positions.shape[0]):
            occupied = set(int(cell) for cell in positions[batch].tolist())
            empty = [int(cell) for cell in self.room.interior.tolist() if int(cell) not in occupied]
            obj = batch % positions.shape[1]
            shifted[batch, obj] = empty[0]
        return shifted


def r440_sensory_loss(output: R440SensoryBridgeOutput, batch: R440SensoryBatch) -> torch.Tensor:
    visible = batch.present.reshape(-1)
    flat_logits = output.position_logits.reshape(-1, output.position_logits.shape[-1])
    flat_positions = batch.positions.reshape(-1)
    if bool(visible.any()):
        position_loss = F.cross_entropy(flat_logits[visible], flat_positions[visible])
    else:
        position_loss = output.position_logits.sum() * 0.0
    presence_loss = F.binary_cross_entropy_with_logits(output.presence_logits, batch.present.float())
    return position_loss + presence_loss


def sensory_predictions(output: R440SensoryBridgeOutput) -> tuple[torch.Tensor, torch.Tensor]:
    return output.position_logits.argmax(dim=-1), output.presence_logits > 0


def exact_visible_layout(predicted: torch.Tensor, present: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    matches = predicted == target
    return torch.where(present, matches, torch.ones_like(matches)).all(dim=-1)


def evaluate_r440_sensory_bridge(
    bridge: R440SensoryBridge,
    renderer: R440SensoryRoomRenderer,
) -> R440SensoryBridgeResult:
    batch = renderer.make_batch()
    deleted = renderer.make_batch(delete_one=True)
    translated_positions = renderer.translate(batch.positions)
    translated = R440SensoryBatch(
        image=renderer.render(translated_positions),
        positions=translated_positions,
        present=torch.ones_like(batch.present, dtype=torch.bool),
    )
    blank = R440SensoryBatch(
        image=torch.zeros_like(batch.image),
        positions=batch.positions,
        present=batch.present,
    )
    with torch.no_grad():
        pred, present = sensory_predictions(bridge(batch.image))
        delete_pred, delete_present = sensory_predictions(bridge(deleted.image))
        translated_pred, translated_present = sensory_predictions(bridge(translated.image))
        blank_pred, blank_present = sensory_predictions(bridge(blank.image))
        ablated = batch.image.clone()
        ablated[:, 2:, :, :] = 0.0
        ablated_pred, ablated_present = sensory_predictions(bridge(ablated))

    position_accuracy = float((pred == batch.positions).float().mean().item())
    exact_layout = float((pred == batch.positions).all(dim=-1).float().mean().item())
    relation_accuracy = float(
        (room_relation_tensor(pred, grid=renderer.room.config.grid) == room_relation_tensor(batch.positions, grid=renderer.room.config.grid))
        .float()
        .mean()
        .item()
    )
    translated_exact = float((translated_pred == translated.positions).all(dim=-1).float().mean().item())
    translated_changed = float((translated.positions != batch.positions).any(dim=-1).float().mean().item())
    delete_visible_exact = float(exact_visible_layout(delete_pred, deleted.present, deleted.positions).float().mean().item())
    delete_presence = float((delete_present == deleted.present).float().mean().item())
    blank_exact = float((blank_pred == blank.positions).all(dim=-1).float().mean().item())
    ablated_exact = float((ablated_pred == batch.positions).all(dim=-1).float().mean().item())
    metrics = {
        "position_accuracy": position_accuracy,
        "exact_layout": exact_layout,
        "relation_accuracy": relation_accuracy,
        "translated_exact_layout": translated_exact,
        "translated_changed_layout": translated_changed,
        "delete_visible_exact_layout": delete_visible_exact,
        "delete_presence_accuracy": delete_presence,
        "blank_exact_layout": blank_exact,
        "object_ablation_exact_layout": ablated_exact,
        "blank_presence_rate": float(blank_present.float().mean().item()),
        "ablated_presence_rate": float(ablated_present.float().mean().item()),
    }
    examples = []
    for idx in range(min(3, renderer.batch_size)):
        examples.append(
            {
                "expected": "/".join(renderer.room.render(batch.positions[idx])),
                "predicted": "/".join(renderer.room.render(pred[idx])),
                "translated_expected": "/".join(renderer.room.render(translated.positions[idx])),
                "translated_predicted": "/".join(renderer.room.render(translated_pred[idx])),
            }
        )
    return R440SensoryBridgeResult(
        seed=renderer.room.seed,
        passed=r440_sensory_bridge_passed(metrics),
        metrics=metrics,
        examples=examples,
    )


def r440_sensory_bridge_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["position_accuracy"] >= 0.95
        and metrics["exact_layout"] >= 0.90
        and metrics["relation_accuracy"] >= 0.95
        and metrics["translated_exact_layout"] >= 0.90
        and metrics["translated_changed_layout"] >= 0.90
        and metrics["delete_visible_exact_layout"] >= 0.90
        and metrics["delete_presence_accuracy"] >= 0.90
        and metrics["blank_exact_layout"] <= 0.35
        and metrics["object_ablation_exact_layout"] <= 0.35
        and metrics["blank_presence_rate"] <= 0.35
        and metrics["ablated_presence_rate"] <= 0.50
    )


def train_r440_sensory_bridge(
    *,
    seed: int = 101,
    batch_size: int = 64,
    steps: int = 240,
    lr: float = 0.006,
) -> tuple[R440SensoryBridge, R440SensoryRoomRenderer, R440SensoryBridgeResult]:
    torch.manual_seed(seed)
    bridge = R440SensoryBridge()
    renderer = R440SensoryRoomRenderer(seed=seed, batch_size=batch_size, core_config=bridge.core_config)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr)
    bridge.train()
    for step in range(steps):
        if step % 7 == 0:
            base = renderer.make_batch()
            batch = R440SensoryBatch(
                image=torch.zeros_like(base.image),
                positions=base.positions,
                present=torch.zeros_like(base.present, dtype=torch.bool),
            )
        else:
            delete_one = step % 5 == 0
            batch = renderer.make_batch(delete_one=delete_one)
        loss = r440_sensory_loss(bridge(batch.image), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    bridge.eval()
    result = evaluate_r440_sensory_bridge(
        bridge,
        R440SensoryRoomRenderer(seed=seed + 9000, batch_size=batch_size, core_config=bridge.core_config),
    )
    return bridge, renderer, result
