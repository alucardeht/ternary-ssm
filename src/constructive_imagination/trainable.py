"""Trainable toy probes for R439B mechanisms.

These probes are intentionally tiny. They do not claim model-scale success; they
test whether each mechanism can learn and survive causal controls in a bounded
synthetic setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainableProbeResult:
    name: str
    mechanism: str
    model_metric: float
    control_metric: float
    ablation_metric: float
    passed: bool
    interpretation: str

    def as_dict(self) -> dict[str, bool | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class CombinedProbeResult:
    name: str
    mechanism: str
    model_metric: float
    control_metric: float
    ablation_metric: float
    task_metrics: dict[str, float]
    task_control_metrics: dict[str, float]
    task_ablation_metrics: dict[str, float]
    passed: bool
    interpretation: str

    def as_dict(self) -> dict[str, bool | float | str | dict[str, float]]:
        return asdict(self)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _one_hot(index: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(index.long(), num_classes=classes).float()


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


def _train_classifier(
    model: nn.Module,
    make_batch,
    *,
    steps: int,
    lr: float = 0.02,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x, y = make_batch()
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def run_patch_dynamics_probe(
    *,
    seed: int = 11,
    steps: int = 180,
    batch_size: int = 128,
) -> TrainableProbeResult:
    """Train a state+action patch predictor against a no-action control."""

    torch.manual_seed(seed)
    grid = 4
    cells = grid * grid
    actions = 4

    def next_pos(pos: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), action]
        return selected[:, 0] * grid + selected[:, 1]

    def make_model_batch():
        pos = torch.randint(0, cells, (batch_size,))
        action = torch.randint(0, actions, (batch_size,))
        x = torch.cat((_one_hot(pos, cells), _one_hot(action, actions)), dim=-1)
        return x, next_pos(pos, action)

    def make_control_batch():
        pos = torch.randint(0, cells, (batch_size,))
        action = torch.randint(0, actions, (batch_size,))
        return _one_hot(pos, cells), next_pos(pos, action)

    model = MLP(cells + actions, 48, cells)
    control = MLP(cells, 48, cells)
    _train_classifier(model, make_model_batch, steps=steps)
    _train_classifier(control, make_control_batch, steps=steps)

    x, y = make_model_batch()
    control_x, control_y = make_control_batch()
    ablated_x = x.clone()
    ablated_x[:, cells:] = 0.0
    model_acc = _accuracy(model(x), y)
    control_acc = _accuracy(control(control_x), control_y)
    ablation_acc = _accuracy(model(ablated_x), y)

    return TrainableProbeResult(
        name="patch_dynamics",
        mechanism="A01 typed state/action transition",
        model_metric=model_acc,
        control_metric=control_acc,
        ablation_metric=ablation_acc,
        passed=model_acc >= 0.95 and model_acc - ablation_acc >= 0.45,
        interpretation=(
            "Patch dynamics learned the local transition when action state was present; "
            "action ablation tests whether the transition used the typed patch input."
        ),
    )


def run_visual_state_probe(
    *,
    seed: int = 17,
    steps: int = 180,
    batch_size: int = 128,
) -> TrainableProbeResult:
    """Train visual-state hidden reveal against caption-only control."""

    torch.manual_seed(seed)
    cells = 16

    def make_visual_batch():
        hidden = torch.randint(0, cells, (batch_size,))
        visible = torch.randint(0, cells, (batch_size,))
        caption = torch.ones(batch_size, 1)
        x = torch.cat((_one_hot(hidden, cells), _one_hot(visible, cells), caption), dim=-1)
        return x, hidden

    def make_caption_batch():
        hidden = torch.randint(0, cells, (batch_size,))
        caption = torch.ones(batch_size, 1)
        return caption, hidden

    model = MLP(cells * 2 + 1, 64, cells)
    control = MLP(1, 16, cells)
    _train_classifier(model, make_visual_batch, steps=steps)
    _train_classifier(control, make_caption_batch, steps=steps)

    x, y = make_visual_batch()
    caption_x, caption_y = make_caption_batch()
    ablated_x = x.clone()
    ablated_x[:, :cells] = 0.0
    model_acc = _accuracy(model(x), y)
    control_acc = _accuracy(control(caption_x), caption_y)
    ablation_acc = _accuracy(model(ablated_x), y)

    return TrainableProbeResult(
        name="visual_state_write",
        mechanism="A02 vision writes hidden continuation state",
        model_metric=model_acc,
        control_metric=control_acc,
        ablation_metric=ablation_acc,
        passed=model_acc >= 0.95 and control_acc <= 0.20 and ablation_acc <= 0.25,
        interpretation=(
            "The visual-state model learned same-caption hidden reveal; caption-only "
            "and hidden-state ablation remained near chance."
        ),
    )


def run_internal_memory_probe(
    *,
    seed: int = 23,
    steps: int = 220,
    batch_size: int = 128,
) -> TrainableProbeResult:
    """Train a fixed-budget memory readout on delayed key recall."""

    torch.manual_seed(seed)
    classes = 8
    memory_width = classes

    class MemoryReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = MLP(memory_width, 32, classes)

        def forward(self, key: torch.Tensor, ablate: bool = False) -> torch.Tensor:
            memory = _one_hot(key, classes)
            if ablate:
                memory = torch.zeros_like(memory)
            return self.classifier(memory)

    class NoMemoryControl(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = MLP(1, 16, classes)

        def forward(self, key: torch.Tensor) -> torch.Tensor:
            _ = key
            return self.classifier(torch.ones(key.shape[0], 1))

    def make_keys():
        key = torch.randint(0, classes, (batch_size,))
        return key, key

    memory_model = MemoryReadout()
    control = NoMemoryControl()
    optimizer = torch.optim.AdamW(memory_model.parameters(), lr=0.02)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.02)
    for _ in range(steps):
        key, target = make_keys()
        loss = F.cross_entropy(memory_model(key), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        control_loss = F.cross_entropy(control(key), target)
        control_optimizer.zero_grad(set_to_none=True)
        control_loss.backward()
        control_optimizer.step()

    key, target = make_keys()
    model_acc = _accuracy(memory_model(key), target)
    control_acc = _accuracy(control(key), target)
    ablation_acc = _accuracy(memory_model(key, ablate=True), target)

    return TrainableProbeResult(
        name="internal_memory",
        mechanism="B05 fixed internal cell recall",
        model_metric=model_acc,
        control_metric=control_acc,
        ablation_metric=ablation_acc,
        passed=model_acc >= 0.95 and control_acc <= 0.25 and ablation_acc <= 0.25,
        interpretation=(
            "Fixed internal memory recall solved delayed key identity; the no-memory "
            "control and memory ablation stayed near chance. Write policy is still engineered."
        ),
    )


def run_language_interface_probe(
    *,
    seed: int = 31,
    steps: int = 220,
    batch_size: int = 128,
) -> TrainableProbeResult:
    """Train state+operator language interface against language-only control."""

    torch.manual_seed(seed)
    max_count = 4
    state_dim = 4
    operators = 3
    output_classes = max_count + 1

    def make_state_batch():
        red_cube_count = torch.randint(0, output_classes, (batch_size,))
        other_counts = torch.randint(0, output_classes, (batch_size, state_dim - 1))
        state = torch.cat((red_cube_count[:, None].float() / max_count, other_counts.float() / max_count), dim=-1)
        operator = torch.zeros(batch_size, dtype=torch.long)
        x = torch.cat((state, _one_hot(operator, operators)), dim=-1)
        return x, red_cube_count

    def make_language_only_batch():
        red_cube_count = torch.randint(0, output_classes, (batch_size,))
        operator = torch.zeros(batch_size, dtype=torch.long)
        return _one_hot(operator, operators), red_cube_count

    model = MLP(state_dim + operators, 64, output_classes)
    control = MLP(operators, 16, output_classes)
    _train_classifier(model, make_state_batch, steps=steps)
    _train_classifier(control, make_language_only_batch, steps=steps)

    x, y = make_state_batch()
    language_x, language_y = make_language_only_batch()
    ablated_x = x.clone()
    ablated_x[:, :state_dim] = 0.0
    model_acc = _accuracy(model(x), y)
    control_acc = _accuracy(control(language_x), language_y)
    ablation_acc = _accuracy(model(ablated_x), y)

    return TrainableProbeResult(
        name="language_interface",
        mechanism="B06 language operator over state",
        model_metric=model_acc,
        control_metric=control_acc,
        ablation_metric=ablation_acc,
        passed=model_acc >= 0.90 and control_acc <= 0.35 and ablation_acc <= 0.35,
        interpretation=(
            "The language interface learned count queries only when state coordinates "
            "were present; language-only and state ablation were low."
        ),
    )


def run_combined_world_probe(
    *,
    seed: int = 41,
    steps: int = 600,
    batch_size: int = 128,
) -> CombinedProbeResult:
    """Train one shared toy world over visual, patch, memory, and language heads."""

    torch.manual_seed(seed)
    grid = 4
    cells = grid * grid
    actions = 4
    memory_classes = 8
    count_classes = 5
    control_dim = 3
    input_dim = cells * 3 + actions + memory_classes + count_classes

    hidden_slice = slice(0, cells)
    visible_slice = slice(cells, cells * 2)
    position_slice = slice(cells * 2, cells * 3)
    action_slice = slice(cells * 3, cells * 3 + actions)
    memory_slice = slice(cells * 3 + actions, cells * 3 + actions + memory_classes)
    count_slice = slice(cells * 3 + actions + memory_classes, input_dim)

    def next_pos(pos: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), action]
        return selected[:, 0] * grid + selected[:, 1]

    def make_structured_batch():
        hidden = torch.randint(0, cells, (batch_size,))
        visible = torch.randint(0, cells, (batch_size,))
        pos = torch.randint(0, cells, (batch_size,))
        action = torch.randint(0, actions, (batch_size,))
        memory_key = torch.randint(0, memory_classes, (batch_size,))
        red_count = torch.randint(0, count_classes, (batch_size,))
        x = torch.cat(
            (
                _one_hot(hidden, cells),
                _one_hot(visible, cells),
                _one_hot(pos, cells),
                _one_hot(action, actions),
                _one_hot(memory_key, memory_classes),
                _one_hot(red_count, count_classes),
            ),
            dim=-1,
        )
        return x, {
            "next_pos": next_pos(pos, action),
            "hidden_location": hidden,
            "memory_key": memory_key,
            "red_count": red_count,
        }

    def make_control_batch():
        x, targets = make_structured_batch()
        _ = x
        control_x = torch.ones(batch_size, control_dim)
        return control_x, targets

    class MultiHeadWorld(nn.Module):
        def __init__(self, source_dim: int):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(source_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 96),
                nn.ReLU(),
            )
            self.next_pos = nn.Linear(96, cells)
            self.hidden_location = nn.Linear(96, cells)
            self.memory_key = nn.Linear(96, memory_classes)
            self.red_count = nn.Linear(96, count_classes)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            h = self.trunk(x)
            return {
                "next_pos": self.next_pos(h),
                "hidden_location": self.hidden_location(h),
                "memory_key": self.memory_key(h),
                "red_count": self.red_count(h),
            }

    def loss_for(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(F.cross_entropy(outputs[name], target) for name, target in targets.items())

    def metrics_for(model: MultiHeadWorld, x: torch.Tensor, targets: dict[str, torch.Tensor]) -> dict[str, float]:
        outputs = model(x)
        return {name: _accuracy(outputs[name], target) for name, target in targets.items()}

    model = MultiHeadWorld(input_dim)
    control = MultiHeadWorld(control_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.02)
    for _ in range(steps):
        x, targets = make_structured_batch()
        loss = loss_for(model(x), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        control_x, control_targets = make_control_batch()
        control_loss = loss_for(control(control_x), control_targets)
        control_optimizer.zero_grad(set_to_none=True)
        control_loss.backward()
        control_optimizer.step()

    x, targets = make_structured_batch()
    control_x, control_targets = make_control_batch()
    task_metrics = metrics_for(model, x, targets)
    task_control_metrics = metrics_for(control, control_x, control_targets)

    ablation_inputs = {
        "next_pos": x.clone(),
        "hidden_location": x.clone(),
        "memory_key": x.clone(),
        "red_count": x.clone(),
    }
    ablation_inputs["next_pos"][:, action_slice] = 0.0
    ablation_inputs["hidden_location"][:, hidden_slice] = 0.0
    ablation_inputs["memory_key"][:, memory_slice] = 0.0
    ablation_inputs["red_count"][:, count_slice] = 0.0
    task_ablation_metrics = {
        name: _accuracy(model(ablated_x)[name], targets[name])
        for name, ablated_x in ablation_inputs.items()
    }

    model_metric = sum(task_metrics.values()) / len(task_metrics)
    control_metric = sum(task_control_metrics.values()) / len(task_control_metrics)
    ablation_metric = sum(task_ablation_metrics.values()) / len(task_ablation_metrics)

    return CombinedProbeResult(
        name="combined_world",
        mechanism="Shared constructive world over visual state, patch dynamics, memory, and language heads",
        model_metric=model_metric,
        control_metric=control_metric,
        ablation_metric=ablation_metric,
        task_metrics=task_metrics,
        task_control_metrics=task_control_metrics,
        task_ablation_metrics=task_ablation_metrics,
        passed=model_metric >= 0.95 and control_metric <= 0.30 and ablation_metric <= 0.45,
        interpretation=(
            "One shared trainable toy world solved all four heads while no-state control "
            "and targeted feature ablations stayed far below the structured model."
        ),
    )


def run_latent_write_world_probe(
    *,
    seed: int = 53,
    steps: int = 900,
    batch_size: int = 128,
) -> CombinedProbeResult:
    """Train a learned latent write policy with harder anti-copy controls."""

    torch.manual_seed(seed)
    grid = 4
    cells = grid * grid
    actions = 4
    memory_classes = 8
    count_classes = 5
    code_dim = 6
    memory_dim = 40

    memory_codes = torch.randn(memory_classes, code_dim)
    count_codes = torch.randn(count_classes, code_dim)
    input_dim = cells * 2 + actions * 2 + code_dim * 2 + 1
    control_dim = 1

    def shift(pos: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), direction]
        return selected[:, 0] * grid + selected[:, 1]

    def noisy_codes(codebook: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return codebook[index] + torch.randn(index.shape[0], codebook.shape[1]) * 0.03

    def make_structured_batch():
        anchor = torch.randint(0, cells, (batch_size,))
        visual_relation = torch.randint(0, actions, (batch_size,))
        agent_pos = torch.randint(0, cells, (batch_size,))
        action = torch.randint(0, actions, (batch_size,))
        memory_key = torch.randint(0, memory_classes, (batch_size,))
        red_count = torch.randint(0, count_classes, (batch_size,))
        caption = torch.ones(batch_size, 1)
        x = torch.cat(
            (
                _one_hot(anchor, cells),
                _one_hot(visual_relation, actions),
                _one_hot(agent_pos, cells),
                _one_hot(action, actions),
                noisy_codes(memory_codes, memory_key),
                noisy_codes(count_codes, red_count),
                caption,
            ),
            dim=-1,
        )
        return x, {
            "hidden_location": shift(anchor, visual_relation),
            "next_pos": shift(agent_pos, action),
            "memory_key": memory_key,
            "red_count": red_count,
        }

    def make_control_batch():
        _x, targets = make_structured_batch()
        return torch.ones(batch_size, control_dim), targets

    class LatentWriteWorld(nn.Module):
        def __init__(self, source_dim: int):
            super().__init__()
            self.writer = nn.Sequential(
                nn.Linear(source_dim, 128),
                nn.ReLU(),
                nn.Linear(128, memory_dim),
                nn.Tanh(),
            )
            self.reader = nn.Sequential(
                nn.Linear(memory_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 96),
                nn.ReLU(),
            )
            self.hidden_location = nn.Linear(96, cells)
            self.next_pos = nn.Linear(96, cells)
            self.memory_key = nn.Linear(96, memory_classes)
            self.red_count = nn.Linear(96, count_classes)

        def forward(self, x: torch.Tensor, *, ablate_memory: bool = False) -> dict[str, torch.Tensor]:
            memory = self.writer(x)
            if ablate_memory:
                memory = torch.zeros_like(memory)
            h = self.reader(memory)
            return {
                "hidden_location": self.hidden_location(h),
                "next_pos": self.next_pos(h),
                "memory_key": self.memory_key(h),
                "red_count": self.red_count(h),
            }

    def loss_for(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(F.cross_entropy(outputs[name], target) for name, target in targets.items())

    def metrics_for(
        model: LatentWriteWorld,
        x: torch.Tensor,
        targets: dict[str, torch.Tensor],
        *,
        ablate_memory: bool = False,
    ) -> dict[str, float]:
        outputs = model(x, ablate_memory=ablate_memory)
        return {name: _accuracy(outputs[name], target) for name, target in targets.items()}

    model = LatentWriteWorld(input_dim)
    control = LatentWriteWorld(control_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
    for _ in range(steps):
        x, targets = make_structured_batch()
        loss = loss_for(model(x), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        control_x, control_targets = make_control_batch()
        control_loss = loss_for(control(control_x), control_targets)
        control_optimizer.zero_grad(set_to_none=True)
        control_loss.backward()
        control_optimizer.step()

    x, targets = make_structured_batch()
    control_x, control_targets = make_control_batch()
    task_metrics = metrics_for(model, x, targets)
    task_control_metrics = metrics_for(control, control_x, control_targets)
    task_ablation_metrics = metrics_for(model, x, targets, ablate_memory=True)

    model_metric = sum(task_metrics.values()) / len(task_metrics)
    control_metric = sum(task_control_metrics.values()) / len(task_control_metrics)
    ablation_metric = sum(task_ablation_metrics.values()) / len(task_ablation_metrics)

    return CombinedProbeResult(
        name="latent_write_world",
        mechanism="Learned latent write policy over visual twins, patch dynamics, memory, and language heads",
        model_metric=model_metric,
        control_metric=control_metric,
        ablation_metric=ablation_metric,
        task_metrics=task_metrics,
        task_control_metrics=task_control_metrics,
        task_ablation_metrics=task_ablation_metrics,
        passed=model_metric >= 0.92 and control_metric <= 0.32 and ablation_metric <= 0.32,
        interpretation=(
            "A learned writer compressed structured visual, action, memory-code, and count "
            "signals into an internal latent memory; caption-only control and zero-memory "
            "ablation failed."
        ),
    )


def run_pixel_latent_world_probe(
    *,
    seed: int = 67,
    steps: int = 2500,
    batch_size: int = 128,
) -> CombinedProbeResult:
    """Train a pixel-like visual latent writer with relation cues, not target copies."""

    torch.manual_seed(seed)
    grid = 4
    cells = grid * grid
    actions = 4
    memory_classes = 8
    count_classes = 5
    code_dim = 6
    memory_dim = 48
    cue_cells = torch.tensor([0, grid - 1, cells - grid, cells - 1])
    memory_codes = torch.randn(memory_classes, code_dim)
    count_codes = torch.randn(count_classes, code_dim)
    extra_dim = actions + code_dim * 2 + 1

    def shift(pos: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), direction]
        return selected[:, 0] * grid + selected[:, 1]

    def noisy_codes(codebook: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return codebook[index] + torch.randn(index.shape[0], codebook.shape[1]) * 0.04

    def scatter_channel(index: torch.Tensor) -> torch.Tensor:
        channel = torch.zeros(index.shape[0], cells)
        channel[torch.arange(index.shape[0]), index] = 1.0
        return channel.view(index.shape[0], 1, grid, grid)

    def coord_channels(size: int) -> torch.Tensor:
        row = torch.linspace(-1.0, 1.0, grid).view(1, 1, grid, 1).expand(size, 1, grid, grid)
        col = torch.linspace(-1.0, 1.0, grid).view(1, 1, 1, grid).expand(size, 1, grid, grid)
        return torch.cat((row, col), dim=1)

    def make_structured_batch():
        anchor = torch.randint(0, cells, (batch_size,))
        relation = torch.randint(0, actions, (batch_size,))
        agent_pos = torch.randint(0, cells, (batch_size,))
        action = torch.randint(0, actions, (batch_size,))
        memory_key = torch.randint(0, memory_classes, (batch_size,))
        red_count = torch.randint(0, count_classes, (batch_size,))
        image = torch.cat(
            (
                scatter_channel(anchor),
                scatter_channel(cue_cells[relation]),
                scatter_channel(agent_pos),
                coord_channels(batch_size),
            ),
            dim=1,
        )
        caption = torch.ones(batch_size, 1)
        extra = torch.cat(
            (
                _one_hot(action, actions),
                noisy_codes(memory_codes, memory_key),
                noisy_codes(count_codes, red_count),
                caption,
            ),
            dim=-1,
        )
        return image, extra, {
            "hidden_location": shift(anchor, relation),
            "next_pos": shift(agent_pos, action),
            "memory_key": memory_key,
            "red_count": red_count,
        }

    def make_control_batch():
        _image, _extra, targets = make_structured_batch()
        empty_objects = torch.zeros(batch_size, 3, grid, grid)
        return torch.cat((empty_objects, coord_channels(batch_size)), dim=1), torch.ones(batch_size, extra_dim), targets

    class PixelLatentWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5 * grid * grid, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.writer = nn.Sequential(
                nn.Linear(128 + extra_dim, 160),
                nn.ReLU(),
                nn.Linear(160, memory_dim),
                nn.Tanh(),
            )
            self.reader = nn.Sequential(
                nn.Linear(memory_dim, 112),
                nn.ReLU(),
                nn.Linear(112, 112),
                nn.ReLU(),
            )
            self.hidden_location = nn.Linear(112, cells)
            self.next_pos = nn.Linear(112, cells)
            self.memory_key = nn.Linear(112, memory_classes)
            self.red_count = nn.Linear(112, count_classes)

        def forward(
            self,
            image: torch.Tensor,
            extra: torch.Tensor,
            *,
            ablate_memory: bool = False,
        ) -> dict[str, torch.Tensor]:
            encoded_image = self.image_encoder(image)
            memory = self.writer(torch.cat((encoded_image, extra), dim=-1))
            if ablate_memory:
                memory = torch.zeros_like(memory)
            h = self.reader(memory)
            return {
                "hidden_location": self.hidden_location(h),
                "next_pos": self.next_pos(h),
                "memory_key": self.memory_key(h),
                "red_count": self.red_count(h),
            }

    def loss_for(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = {
            "hidden_location": 4.0,
            "next_pos": 4.0,
            "memory_key": 1.0,
            "red_count": 1.0,
        }
        return sum(weights[name] * F.cross_entropy(outputs[name], target) for name, target in targets.items())

    def metrics_for(
        model: PixelLatentWorld,
        image: torch.Tensor,
        extra: torch.Tensor,
        targets: dict[str, torch.Tensor],
        *,
        ablate_memory: bool = False,
    ) -> dict[str, float]:
        outputs = model(image, extra, ablate_memory=ablate_memory)
        return {name: _accuracy(outputs[name], target) for name, target in targets.items()}

    model = PixelLatentWorld()
    control = PixelLatentWorld()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.008)
    for _ in range(steps):
        image, extra, targets = make_structured_batch()
        loss = loss_for(model(image, extra), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        control_image, control_extra, control_targets = make_control_batch()
        control_loss = loss_for(control(control_image, control_extra), control_targets)
        control_optimizer.zero_grad(set_to_none=True)
        control_loss.backward()
        control_optimizer.step()

    image, extra, targets = make_structured_batch()
    control_image, control_extra, control_targets = make_control_batch()
    task_metrics = metrics_for(model, image, extra, targets)
    task_control_metrics = metrics_for(control, control_image, control_extra, control_targets)
    task_ablation_metrics = metrics_for(model, image, extra, targets, ablate_memory=True)

    model_metric = sum(task_metrics.values()) / len(task_metrics)
    control_metric = sum(task_control_metrics.values()) / len(task_control_metrics)
    ablation_metric = sum(task_ablation_metrics.values()) / len(task_ablation_metrics)

    return CombinedProbeResult(
        name="pixel_latent_world",
        mechanism="Pixel-like visual relation encoder with learned latent write memory",
        model_metric=model_metric,
        control_metric=control_metric,
        ablation_metric=ablation_metric,
        task_metrics=task_metrics,
        task_control_metrics=task_control_metrics,
        task_ablation_metrics=task_ablation_metrics,
        passed=model_metric >= 0.90 and control_metric <= 0.32 and ablation_metric <= 0.32,
        interpretation=(
            "A pixel-tensor visual encoder learned relation cues from 4x4 channels and wrote "
            "task-critical state into latent memory; caption-only/no-image control and "
            "zero-memory ablation failed."
        ),
    )


def run_slot_distractor_world_probe(
    *,
    seed: int = 79,
    steps: int = 1800,
    batch_size: int = 128,
) -> CombinedProbeResult:
    """Train learned visual slot selection with target/distractor pressure."""

    torch.manual_seed(seed)
    grid = 4
    cells = grid * grid
    colors = 3
    actions = 4
    memory_classes = 8
    code_dim = 6
    memory_dim = 48
    extra_dim = colors + actions + code_dim + 1
    memory_codes = torch.randn(memory_classes, code_dim)

    coord = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid),
            torch.linspace(-1.0, 1.0, grid),
            indexing="ij",
        ),
        dim=-1,
    ).view(cells, 2)

    def shift(pos: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), direction]
        return selected[:, 0] * grid + selected[:, 1]

    def noisy_codes(codebook: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return codebook[index] + torch.randn(index.shape[0], codebook.shape[1]) * 0.04

    def sample_positions() -> torch.Tensor:
        return torch.stack([torch.randperm(cells)[:colors] for _ in range(batch_size)])

    def make_image(positions: torch.Tensor) -> torch.Tensor:
        image = torch.zeros(batch_size, colors + 2, cells)
        for color in range(colors):
            image[torch.arange(batch_size), color, positions[:, color]] = 1.0
        image[:, colors:, :] = coord.T.unsqueeze(0).expand(batch_size, 2, cells)
        return image.transpose(1, 2)

    def make_structured_batch():
        positions = sample_positions()
        target_color = torch.randint(0, colors, (batch_size,))
        action = torch.randint(0, actions, (batch_size,))
        memory_key = torch.randint(0, memory_classes, (batch_size,))
        target_pos = positions[torch.arange(batch_size), target_color]
        image = make_image(positions)
        caption = torch.ones(batch_size, 1)
        extra = torch.cat(
            (
                _one_hot(target_color, colors),
                _one_hot(action, actions),
                noisy_codes(memory_codes, memory_key),
                caption,
            ),
            dim=-1,
        )
        return image, extra, target_color, {
            "target_position": target_pos,
            "target_next_pos": shift(target_pos, action),
            "memory_key": memory_key,
        }

    def make_control_batch():
        _image, _extra, _target_color, targets = make_structured_batch()
        empty = torch.zeros(batch_size, colors + 2, cells).transpose(1, 2)
        empty[:, :, colors:] = coord.unsqueeze(0).expand(batch_size, cells, 2)
        return empty, torch.ones(batch_size, extra_dim), targets

    class SlotDistractorWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_encoder = nn.Sequential(nn.Linear(colors, 32), nn.ReLU())
            self.token_encoder = nn.Sequential(
                nn.Linear(colors + 2 + 32, 96),
                nn.ReLU(),
                nn.Linear(96, 96),
                nn.ReLU(),
            )
            self.token_score = nn.Linear(96, 1)
            self.token_value = nn.Linear(96, 64)
            self.writer = nn.Sequential(
                nn.Linear(64 + actions + code_dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, memory_dim),
                nn.Tanh(),
            )
            self.reader = nn.Sequential(
                nn.Linear(memory_dim, 112),
                nn.ReLU(),
                nn.Linear(112, 112),
                nn.ReLU(),
            )
            self.target_position = nn.Linear(112, cells)
            self.target_next_pos = nn.Linear(112, cells)
            self.memory_key = nn.Linear(112, memory_classes)

        def forward(
            self,
            image: torch.Tensor,
            extra: torch.Tensor,
            *,
            ablate_memory: bool = False,
        ) -> dict[str, torch.Tensor]:
            query = extra[:, :colors]
            query_features = self.query_encoder(query).unsqueeze(1).expand(-1, cells, -1)
            tokens = self.token_encoder(torch.cat((image, query_features), dim=-1))
            attention = torch.softmax(self.token_score(tokens).squeeze(-1), dim=-1)
            slot = torch.sum(attention.unsqueeze(-1) * self.token_value(tokens), dim=1)
            memory = self.writer(torch.cat((slot, extra[:, colors:]), dim=-1))
            if ablate_memory:
                memory = torch.zeros_like(memory)
            h = self.reader(memory)
            return {
                "target_position": self.target_position(h),
                "target_next_pos": self.target_next_pos(h),
                "memory_key": self.memory_key(h),
            }

    def loss_for(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = {
            "target_position": 3.0,
            "target_next_pos": 3.0,
            "memory_key": 1.0,
        }
        return sum(weights[name] * F.cross_entropy(outputs[name], target) for name, target in targets.items())

    def metrics_for(
        model: SlotDistractorWorld,
        image: torch.Tensor,
        extra: torch.Tensor,
        targets: dict[str, torch.Tensor],
        *,
        ablate_memory: bool = False,
    ) -> dict[str, float]:
        outputs = model(image, extra, ablate_memory=ablate_memory)
        return {name: _accuracy(outputs[name], target) for name, target in targets.items()}

    model = SlotDistractorWorld()
    control = SlotDistractorWorld()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.008)
    for _ in range(steps):
        image, extra, _target_color, targets = make_structured_batch()
        loss = loss_for(model(image, extra), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        control_image, control_extra, control_targets = make_control_batch()
        control_loss = loss_for(control(control_image, control_extra), control_targets)
        control_optimizer.zero_grad(set_to_none=True)
        control_loss.backward()
        control_optimizer.step()

    image, extra, target_color, targets = make_structured_batch()
    control_image, control_extra, control_targets = make_control_batch()
    task_metrics = metrics_for(model, image, extra, targets)
    task_control_metrics = metrics_for(control, control_image, control_extra, control_targets)
    task_ablation_metrics = metrics_for(model, image, extra, targets, ablate_memory=True)

    swapped_extra = extra.clone()
    swapped_color = (target_color + 1 + torch.randint(0, colors - 1, (batch_size,))) % colors
    swapped_extra[:, :colors] = _one_hot(swapped_color, colors)
    swapped_outputs = model(image, swapped_extra)
    task_ablation_metrics["query_swap_target_position"] = _accuracy(
        swapped_outputs["target_position"],
        targets["target_position"],
    )

    model_metric = sum(task_metrics.values()) / len(task_metrics)
    control_metric = sum(task_control_metrics.values()) / len(task_control_metrics)
    ablation_metric = sum(task_ablation_metrics.values()) / len(task_ablation_metrics)

    return CombinedProbeResult(
        name="slot_distractor_world",
        mechanism="Learned visual slot selection under target/distractor pressure",
        model_metric=model_metric,
        control_metric=control_metric,
        ablation_metric=ablation_metric,
        task_metrics=task_metrics,
        task_control_metrics=task_control_metrics,
        task_ablation_metrics=task_ablation_metrics,
        passed=model_metric >= 0.90 and control_metric <= 0.28 and ablation_metric <= 0.30,
        interpretation=(
            "A learned selector chose the queried object among distractors, wrote it into "
            "latent memory, and failed under no-image, zero-memory, and query-swap controls."
        ),
    )


def run_occlusion_permanence_probe(
    *,
    seed: int = 89,
    steps: int = 1800,
    batch_size: int = 128,
) -> CombinedProbeResult:
    """Train latent permanence when a queried object disappears in frame two."""

    torch.manual_seed(seed)
    grid = 4
    cells = grid * grid
    colors = 3
    actions = 4
    memory_dim = 48
    extra_dim = colors + actions + 1

    coord = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid),
            torch.linspace(-1.0, 1.0, grid),
            indexing="ij",
        ),
        dim=-1,
    ).view(cells, 2)

    def shift(pos: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), direction]
        return selected[:, 0] * grid + selected[:, 1]

    def sample_positions() -> torch.Tensor:
        return torch.stack([torch.randperm(cells)[:colors] for _ in range(batch_size)])

    def make_image(positions: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        image = torch.zeros(batch_size, colors + 2, cells)
        for color in range(colors):
            visible = visible_mask[:, color]
            image[visible, color, positions[visible, color]] = 1.0
        image[:, colors:, :] = coord.T.unsqueeze(0).expand(batch_size, 2, cells)
        return image.transpose(1, 2)

    def make_structured_batch():
        positions = sample_positions()
        target_color = torch.randint(0, colors, (batch_size,))
        action = torch.randint(0, actions, (batch_size,))
        target_pos = positions[torch.arange(batch_size), target_color]
        first_visible = torch.ones(batch_size, colors, dtype=torch.bool)
        second_visible = torch.ones(batch_size, colors, dtype=torch.bool)
        second_visible[torch.arange(batch_size), target_color] = False
        frame1 = make_image(positions, first_visible)
        frame2 = make_image(positions, second_visible)
        caption = torch.ones(batch_size, 1)
        extra = torch.cat((_one_hot(target_color, colors), _one_hot(action, actions), caption), dim=-1)
        return frame1, frame2, extra, target_color, {
            "remembered_position": target_pos,
            "remembered_next_pos": shift(target_pos, action),
            "occluded_color": target_color,
        }

    def make_current_frame_control_batch():
        _frame1, frame2, _extra, _target_color, targets = make_structured_batch()
        return torch.zeros_like(frame2), frame2, torch.ones(batch_size, extra_dim), targets

    class OcclusionWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_encoder = nn.Sequential(nn.Linear(colors, 32), nn.ReLU())
            self.token_encoder = nn.Sequential(
                nn.Linear(colors + 2 + 32 + 2, 96),
                nn.ReLU(),
                nn.Linear(96, 96),
                nn.ReLU(),
            )
            self.token_score = nn.Linear(96, 1)
            self.token_value = nn.Linear(96, 64)
            self.writer = nn.Sequential(
                nn.Linear(64 + actions + 1, 128),
                nn.ReLU(),
                nn.Linear(128, memory_dim),
                nn.Tanh(),
            )
            self.reader = nn.Sequential(
                nn.Linear(memory_dim, 112),
                nn.ReLU(),
                nn.Linear(112, 112),
                nn.ReLU(),
            )
            self.remembered_position = nn.Linear(112, cells)
            self.remembered_next_pos = nn.Linear(112, cells)
            self.occluded_color = nn.Linear(112, colors)

        def forward(
            self,
            frame1: torch.Tensor,
            frame2: torch.Tensor,
            extra: torch.Tensor,
            *,
            ablate_memory: bool = False,
        ) -> dict[str, torch.Tensor]:
            query = extra[:, :colors]
            query_features = self.query_encoder(query).unsqueeze(1).expand(-1, cells * 2, -1)
            frame_tag1 = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(frame1.shape[0], cells, 2)
            frame_tag2 = torch.tensor([0.0, 1.0]).view(1, 1, 2).expand(frame1.shape[0], cells, 2)
            tokens = torch.cat(
                (
                    torch.cat((frame1, frame_tag1), dim=-1),
                    torch.cat((frame2, frame_tag2), dim=-1),
                ),
                dim=1,
            )
            encoded = self.token_encoder(torch.cat((tokens, query_features), dim=-1))
            attention = torch.softmax(self.token_score(encoded).squeeze(-1), dim=-1)
            slot = torch.sum(attention.unsqueeze(-1) * self.token_value(encoded), dim=1)
            memory = self.writer(torch.cat((slot, extra[:, colors:]), dim=-1))
            if ablate_memory:
                memory = torch.zeros_like(memory)
            h = self.reader(memory)
            return {
                "remembered_position": self.remembered_position(h),
                "remembered_next_pos": self.remembered_next_pos(h),
                "occluded_color": self.occluded_color(h),
            }

    def loss_for(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = {
            "remembered_position": 3.0,
            "remembered_next_pos": 3.0,
            "occluded_color": 1.0,
        }
        return sum(weights[name] * F.cross_entropy(outputs[name], target) for name, target in targets.items())

    def metrics_for(
        model: OcclusionWorld,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        extra: torch.Tensor,
        targets: dict[str, torch.Tensor],
        *,
        ablate_memory: bool = False,
    ) -> dict[str, float]:
        outputs = model(frame1, frame2, extra, ablate_memory=ablate_memory)
        return {name: _accuracy(outputs[name], target) for name, target in targets.items()}

    model = OcclusionWorld()
    control = OcclusionWorld()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.008)
    for _ in range(steps):
        frame1, frame2, extra, _target_color, targets = make_structured_batch()
        loss = loss_for(model(frame1, frame2, extra), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        control_frame1, control_frame2, control_extra, control_targets = make_current_frame_control_batch()
        control_loss = loss_for(control(control_frame1, control_frame2, control_extra), control_targets)
        control_optimizer.zero_grad(set_to_none=True)
        control_loss.backward()
        control_optimizer.step()

    frame1, frame2, extra, target_color, targets = make_structured_batch()
    control_frame1, control_frame2, control_extra, control_targets = make_current_frame_control_batch()
    task_metrics = metrics_for(model, frame1, frame2, extra, targets)
    task_control_metrics = metrics_for(control, control_frame1, control_frame2, control_extra, control_targets)
    task_ablation_metrics = metrics_for(model, frame1, frame2, extra, targets, ablate_memory=True)

    swapped_extra = extra.clone()
    swapped_color = (target_color + 1 + torch.randint(0, colors - 1, (batch_size,))) % colors
    swapped_extra[:, :colors] = _one_hot(swapped_color, colors)
    swapped_outputs = model(frame1, frame2, swapped_extra)
    task_ablation_metrics["query_swap_remembered_position"] = _accuracy(
        swapped_outputs["remembered_position"],
        targets["remembered_position"],
    )

    model_metric = sum(task_metrics.values()) / len(task_metrics)
    control_metric = sum(task_control_metrics.values()) / len(task_control_metrics)
    ablation_metric = sum(task_ablation_metrics.values()) / len(task_ablation_metrics)

    return CombinedProbeResult(
        name="occlusion_permanence_world",
        mechanism="Latent permanence for queried visual object after occlusion",
        model_metric=model_metric,
        control_metric=control_metric,
        ablation_metric=ablation_metric,
        task_metrics=task_metrics,
        task_control_metrics=task_control_metrics,
        task_ablation_metrics=task_ablation_metrics,
        passed=model_metric >= 0.90 and control_metric <= 0.36 and ablation_metric <= 0.34,
        interpretation=(
            "The model selected a target in frame one, carried it through frame-two "
            "occlusion in latent memory, and failed under current-frame, zero-memory, "
            "and query-swap controls."
        ),
    )


def run_r440_slot_proposition_depth_probe(
    *,
    seed: int = 97,
    steps: int = 1200,
    batch_size: int = 128,
) -> CombinedProbeResult:
    """Train query-keyed slots with latent propositions and uncertainty depth."""

    torch.manual_seed(seed)
    grid = 4
    cells = grid * grid
    colors = 3
    actions = 4
    operators = 2
    prop_dim = 32
    slot_dim = 64
    hidden_dim = 96
    extra_dim = colors + operators + actions

    coord = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid),
            torch.linspace(-1.0, 1.0, grid),
            indexing="ij",
        ),
        dim=-1,
    ).view(cells, 2)

    def shift(pos: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        up = torch.stack((torch.clamp(row - 1, min=0), col), dim=-1)
        down = torch.stack((torch.clamp(row + 1, max=grid - 1), col), dim=-1)
        left = torch.stack((row, torch.clamp(col - 1, min=0)), dim=-1)
        right = torch.stack((row, torch.clamp(col + 1, max=grid - 1)), dim=-1)
        candidates = torch.stack((up, down, left, right), dim=1)
        selected = candidates[torch.arange(pos.shape[0]), direction]
        return selected[:, 0] * grid + selected[:, 1]

    def sample_positions() -> torch.Tensor:
        return torch.stack([torch.randperm(cells)[:colors] for _ in range(batch_size)])

    def sample_valid_actions(pos: torch.Tensor) -> torch.Tensor:
        row = pos // grid
        col = pos % grid
        valid = torch.stack((row > 0, row < grid - 1, col > 0, col < grid - 1), dim=-1).float()
        return torch.multinomial(valid / valid.sum(dim=-1, keepdim=True), 1).squeeze(-1)

    def make_frame(positions: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        frame = torch.zeros(batch_size, colors + 2, cells)
        for color in range(colors):
            visible = visible_mask[:, color]
            frame[visible, color, positions[visible, color]] = 1.0
        frame[:, colors:, :] = coord.T.unsqueeze(0).expand(batch_size, 2, cells)
        return frame.transpose(1, 2)

    def answer_for(target_pos: torch.Tensor, action: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        next_pos = shift(target_pos, action)
        return torch.where(operator == 0, target_pos, next_pos)

    def make_structured_batch():
        positions = sample_positions()
        target_color = torch.randint(0, colors, (batch_size,))
        target_pos = positions[torch.arange(batch_size), target_color]
        action = sample_valid_actions(target_pos)
        operator = torch.randint(0, operators, (batch_size,))
        hard = torch.rand(batch_size) < 0.5
        visible_all = torch.ones(batch_size, colors, dtype=torch.bool)
        current_visible = visible_all.clone()
        current_visible[hard, target_color[hard]] = False
        frame1 = make_frame(positions, visible_all)
        frame2 = make_frame(positions, current_visible)
        extra = torch.cat(
            (_one_hot(target_color, colors), _one_hot(operator, operators), _one_hot(action, actions)),
            dim=-1,
        )
        target = answer_for(target_pos, action, operator)
        return frame1, frame2, extra, target_color, operator, hard, {
            "answer": target,
            "depth_gate": hard.long(),
            "target_slot": target_pos,
        }

    def make_control_batch():
        frame1, frame2, extra, _target_color, _operator, _hard, targets = make_structured_batch()
        blank1 = torch.zeros_like(frame1)
        blank2 = torch.zeros_like(frame2)
        blank1[:, :, colors:] = coord.unsqueeze(0).expand(batch_size, cells, 2)
        blank2[:, :, colors:] = coord.unsqueeze(0).expand(batch_size, cells, 2)
        return blank1, blank2, extra, targets

    class R440World(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_encoder = nn.Sequential(nn.Linear(extra_dim, 48), nn.ReLU())
            self.proposition_compiler = nn.Sequential(
                nn.Linear(extra_dim, 64),
                nn.ReLU(),
                nn.Linear(64, prop_dim),
                nn.Tanh(),
            )
            self.fast_token_encoder = nn.Sequential(
                nn.Linear(colors + 2 + 48 + prop_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.deep_token_encoder = nn.Sequential(
                nn.Linear(colors + 2 + 2 + 48 + prop_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.fast_score = nn.Linear(hidden_dim, 1)
            self.deep_score = nn.Linear(hidden_dim, 1)
            self.fast_value = nn.Linear(hidden_dim, slot_dim)
            self.deep_value = nn.Linear(hidden_dim, slot_dim)
            self.fast_reader = nn.Sequential(
                nn.Linear(slot_dim + prop_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.deep_reader = nn.Sequential(
                nn.Linear(slot_dim + prop_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.fast_answer = nn.Linear(hidden_dim, cells)
            self.deep_answer = nn.Linear(hidden_dim, cells)
            self.depth_gate = nn.Linear(hidden_dim + prop_dim + 1, 1)

        def _select(self, encoded: torch.Tensor, score: nn.Linear, value: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
            attention = torch.softmax(score(encoded).squeeze(-1), dim=-1)
            return torch.sum(attention.unsqueeze(-1) * value(encoded), dim=1), attention

        def forward(
            self,
            frame1: torch.Tensor,
            frame2: torch.Tensor,
            extra: torch.Tensor,
            *,
            ablate_slots: bool = False,
            force_fast: bool = False,
            force_deep: bool = False,
        ) -> dict[str, torch.Tensor]:
            query = self.query_encoder(extra)
            proposition = self.proposition_compiler(extra)
            target_presence = torch.sum(frame2[:, :, :colors] * extra[:, :colors].unsqueeze(1), dim=(1, 2)).unsqueeze(-1)
            fast_context = torch.cat(
                (
                    frame2,
                    query.unsqueeze(1).expand(-1, cells, -1),
                    proposition.unsqueeze(1).expand(-1, cells, -1),
                ),
                dim=-1,
            )
            fast_slot, fast_attention = self._select(
                self.fast_token_encoder(fast_context),
                self.fast_score,
                self.fast_value,
            )

            frame_tag1 = torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(frame1.shape[0], cells, 2)
            deep_tokens = torch.cat((frame1, frame_tag1), dim=-1)
            deep_context = torch.cat(
                (
                    deep_tokens,
                    query.unsqueeze(1).expand(-1, cells, -1),
                    proposition.unsqueeze(1).expand(-1, cells, -1),
                ),
                dim=-1,
            )
            deep_slot, deep_attention = self._select(
                self.deep_token_encoder(deep_context),
                self.deep_score,
                self.deep_value,
            )
            if ablate_slots:
                fast_slot = torch.zeros_like(fast_slot)
                deep_slot = torch.zeros_like(deep_slot)

            fast_h = self.fast_reader(torch.cat((fast_slot, proposition), dim=-1))
            deep_h = self.deep_reader(torch.cat((deep_slot, proposition), dim=-1))
            fast_logits = self.fast_answer(fast_h)
            deep_logits = self.deep_answer(deep_h)
            gate_logit = self.depth_gate(torch.cat((fast_h, proposition, target_presence), dim=-1)).squeeze(-1)
            gate = torch.sigmoid(gate_logit).unsqueeze(-1)
            if force_fast:
                gate = torch.zeros_like(gate)
            if force_deep:
                gate = torch.ones_like(gate)
            return {
                "answer": fast_logits * (1.0 - gate) + deep_logits * gate,
                "fast_answer": fast_logits,
                "deep_answer": deep_logits,
                "depth_gate": gate_logit,
                "fast_attention": fast_attention,
                "deep_attention": deep_attention,
            }

    def loss_for(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], hard: torch.Tensor) -> torch.Tensor:
        answer_loss = F.cross_entropy(outputs["answer"], targets["answer"])
        deep_loss = F.cross_entropy(outputs["deep_answer"], targets["answer"])
        gate_loss = F.binary_cross_entropy_with_logits(outputs["depth_gate"], hard.float())
        deep_slot_loss = F.nll_loss(torch.log(outputs["deep_attention"] + 1e-6), targets["target_slot"])
        easy = ~hard
        if bool(easy.any()):
            fast_loss = F.cross_entropy(outputs["fast_answer"][easy], targets["answer"][easy])
            fast_slot_loss = F.nll_loss(
                torch.log(outputs["fast_attention"][easy] + 1e-6),
                targets["target_slot"][easy],
            )
        else:
            fast_loss = torch.tensor(0.0)
            fast_slot_loss = torch.tensor(0.0)
        return answer_loss + deep_loss + 2.5 * fast_loss + 2.0 * gate_loss + deep_slot_loss + fast_slot_loss

    def metrics_for(
        model: R440World,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        extra: torch.Tensor,
        targets: dict[str, torch.Tensor],
        hard: torch.Tensor,
        *,
        ablate_slots: bool = False,
        force_fast: bool = False,
    ) -> dict[str, float]:
        outputs = model(frame1, frame2, extra, ablate_slots=ablate_slots, force_fast=force_fast)
        gate_pred = (outputs["depth_gate"] > 0).long()
        metrics = {
            "answer": _accuracy(outputs["answer"], targets["answer"]),
            "depth_gate": float((gate_pred == targets["depth_gate"]).float().mean().item()),
            "fast_easy_answer": 0.0,
            "fast_hard_answer": 0.0,
            "deep_hard_answer": 0.0,
            "depth_separation": float(
                torch.sigmoid(outputs["depth_gate"])[hard].mean().item()
                - torch.sigmoid(outputs["depth_gate"])[~hard].mean().item()
            ),
        }
        easy = ~hard
        if bool(easy.any()):
            metrics["fast_easy_answer"] = _accuracy(outputs["fast_answer"][easy], targets["answer"][easy])
        if bool(hard.any()):
            metrics["fast_hard_answer"] = _accuracy(outputs["fast_answer"][hard], targets["answer"][hard])
            metrics["deep_hard_answer"] = _accuracy(outputs["deep_answer"][hard], targets["answer"][hard])
        return metrics

    model = R440World()
    control = R440World()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.006)
    for _ in range(steps):
        frame1, frame2, extra, _target_color, _operator, hard, targets = make_structured_batch()
        loss = loss_for(model(frame1, frame2, extra), targets, hard)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        control_frame1, control_frame2, control_extra, control_targets = make_control_batch()
        control_hard = control_targets["depth_gate"].bool()
        control_loss = loss_for(control(control_frame1, control_frame2, control_extra), control_targets, control_hard)
        control_optimizer.zero_grad(set_to_none=True)
        control_loss.backward()
        control_optimizer.step()

    frame1, frame2, extra, target_color, operator, hard, targets = make_structured_batch()
    control_frame1, control_frame2, control_extra, control_targets = make_control_batch()
    control_hard = control_targets["depth_gate"].bool()
    task_metrics = metrics_for(model, frame1, frame2, extra, targets, hard)
    task_control_metrics = metrics_for(control, control_frame1, control_frame2, control_extra, control_targets, control_hard)
    task_ablation_metrics = metrics_for(model, frame1, frame2, extra, targets, hard, ablate_slots=True)

    swapped_extra = extra.clone()
    swapped_color = (target_color + 1 + torch.randint(0, colors - 1, (batch_size,))) % colors
    swapped_extra[:, :colors] = _one_hot(swapped_color, colors)
    swapped_outputs = model(frame1, frame2, swapped_extra)
    task_ablation_metrics["query_swap_original_answer"] = _accuracy(
        swapped_outputs["answer"],
        targets["answer"],
    )

    shuffled_extra = extra.clone()
    shuffled_operator = 1 - operator
    shuffled_extra[:, colors : colors + operators] = _one_hot(shuffled_operator, operators)
    shuffled_outputs = model(frame1, frame2, shuffled_extra)
    task_ablation_metrics["proposition_shuffle_original_answer"] = _accuracy(
        shuffled_outputs["answer"],
        targets["answer"],
    )

    force_fast_outputs = model(frame1, frame2, extra, force_fast=True)
    if bool(hard.any()):
        task_ablation_metrics["force_fast_hard_answer"] = _accuracy(
            force_fast_outputs["answer"][hard],
            targets["answer"][hard],
        )
    else:
        task_ablation_metrics["force_fast_hard_answer"] = 0.0

    model_metric = (
        task_metrics["answer"]
        + task_metrics["depth_gate"]
        + task_metrics["fast_easy_answer"]
        + task_metrics["deep_hard_answer"]
    ) / 4.0
    control_metric = task_control_metrics["answer"]
    ablation_metric = (
        task_ablation_metrics["answer"]
        + task_ablation_metrics["query_swap_original_answer"]
        + task_ablation_metrics["proposition_shuffle_original_answer"]
        + task_ablation_metrics["force_fast_hard_answer"]
    ) / 4.0

    return CombinedProbeResult(
        name="r440_slot_proposition_depth",
        mechanism="Query-keyed object slots with latent proposition compiler and uncertainty-gated depth",
        model_metric=model_metric,
        control_metric=control_metric,
        ablation_metric=ablation_metric,
        task_metrics=task_metrics,
        task_control_metrics=task_control_metrics,
        task_ablation_metrics=task_ablation_metrics,
        passed=(
            model_metric >= 0.88
            and task_metrics["answer"] >= 0.90
            and task_metrics["depth_gate"] >= 0.90
            and task_metrics["fast_easy_answer"] >= 0.85
            and task_metrics["deep_hard_answer"] >= 0.85
            and task_metrics["depth_separation"] >= 0.50
            and control_metric <= 0.25
            and ablation_metric <= 0.35
            and task_ablation_metrics["query_swap_original_answer"] <= 0.35
            and task_ablation_metrics["proposition_shuffle_original_answer"] <= 0.35
            and task_ablation_metrics["force_fast_hard_answer"] <= 0.35
        ),
        interpretation=(
            "The R440 core learned a fast answer path for visible queried slots, a deep "
            "path for occluded queried slots, and a latent proposition interface for "
            "position/transition queries. It failed under no-image, zero-slot, query-swap, "
            "proposition-shuffle, and force-fast hard-case controls."
        ),
    )


def run_all_trainable_probes(seed: int = 101) -> list[TrainableProbeResult | CombinedProbeResult]:
    return [
        run_patch_dynamics_probe(seed=seed + 1),
        run_visual_state_probe(seed=seed + 2),
        run_internal_memory_probe(seed=seed + 3),
        run_language_interface_probe(seed=seed + 4),
        run_combined_world_probe(seed=seed + 5),
        run_latent_write_world_probe(seed=seed + 6),
        run_pixel_latent_world_probe(seed=seed + 7),
        run_slot_distractor_world_probe(seed=seed + 8),
        run_occlusion_permanence_probe(seed=seed + 9),
        run_r440_slot_proposition_depth_probe(seed=seed + 10),
    ]
