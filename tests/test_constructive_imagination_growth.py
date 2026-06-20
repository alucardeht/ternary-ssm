import unittest

import torch
from torch.nn import functional as F

from src.constructive_imagination import (
    ConstructiveCognitiveCore,
    ConstructiveCoreConfig,
    profile_constructive_core,
)


def _one_hot(index: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(index.long(), num_classes=classes).float()


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


class R440BatchFactory:
    def __init__(self, *, seed: int, batch_size: int = 128):
        torch.manual_seed(seed)
        self.batch_size = batch_size
        self.config = ConstructiveCoreConfig()
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
        frame = torch.zeros(self.batch_size, cfg.colors + 2, cfg.cells)
        for color in range(cfg.colors):
            visible = visible_mask[:, color]
            frame[visible, color, positions[visible, color]] = 1.0
        frame[:, cfg.colors :, :] = self.coord.T.unsqueeze(0).expand(self.batch_size, 2, cfg.cells)
        return frame.transpose(1, 2)

    def make_batch(self):
        cfg = self.config
        positions = self.sample_positions()
        target_color = torch.randint(0, cfg.colors, (self.batch_size,))
        target_pos = positions[torch.arange(self.batch_size), target_color]
        action = self.sample_valid_actions(target_pos)
        operator = torch.randint(0, cfg.operators, (self.batch_size,))
        hard = torch.rand(self.batch_size) < 0.5
        visible_all = torch.ones(self.batch_size, cfg.colors, dtype=torch.bool)
        current_visible = visible_all.clone()
        current_visible[hard, target_color[hard]] = False
        frame1 = self.make_frame(positions, visible_all)
        frame2 = self.make_frame(positions, current_visible)
        extra = torch.cat(
            (_one_hot(target_color, cfg.colors), _one_hot(operator, cfg.operators), _one_hot(action, cfg.actions)),
            dim=-1,
        )
        next_pos = self.shift(target_pos, action)
        answer = torch.where(operator == 0, target_pos, next_pos)
        return frame1, frame2, extra, target_color, operator, hard, {
            "answer": answer,
            "depth_gate": hard.long(),
            "target_slot": target_pos,
        }

    def make_no_image_batch(self):
        frame1, frame2, extra, _target_color, _operator, _hard, targets = self.make_batch()
        blank1 = torch.zeros_like(frame1)
        blank2 = torch.zeros_like(frame2)
        blank1[:, :, self.config.colors :] = self.coord.unsqueeze(0).expand(self.batch_size, self.config.cells, 2)
        blank2[:, :, self.config.colors :] = self.coord.unsqueeze(0).expand(self.batch_size, self.config.cells, 2)
        return blank1, blank2, extra, targets


def _loss_for(outputs, targets: dict[str, torch.Tensor], hard: torch.Tensor) -> torch.Tensor:
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


def _train_core(seed: int = 515, steps: int = 900) -> tuple[ConstructiveCognitiveCore, R440BatchFactory]:
    factory = R440BatchFactory(seed=seed)
    model = ConstructiveCognitiveCore(factory.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    for _ in range(steps):
        frame1, frame2, extra, _target_color, _operator, hard, targets = factory.make_batch()
        loss = _loss_for(model(frame1, frame2, extra), targets, hard)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, factory


class ConstructiveImaginationGrowthTests(unittest.TestCase):
    def test_core_outputs_and_cost_profile_are_explicit(self):
        factory = R440BatchFactory(seed=11, batch_size=16)
        model = ConstructiveCognitiveCore(factory.config)
        frame1, frame2, extra, _target_color, _operator, _hard, _targets = factory.make_batch()

        output = model(frame1, frame2, extra)
        fast = model.forward_fast(frame2, extra)
        deep = model.forward_deep(frame1, extra)
        profile = profile_constructive_core(model, route_probability=float(output.route_rate.item()))

        self.assertEqual(output.answer.shape, (16, factory.config.cells))
        self.assertEqual(fast["attention"].shape, (16, factory.config.cells))
        self.assertEqual(deep["attention"].shape, (16, factory.config.cells))
        self.assertGreater(profile.total_params, 0)
        self.assertGreater(profile.deep_path_params, 0)
        self.assertGreater(profile.fast_path_params, 0)
        self.assertLess(profile.expected_active_params, profile.total_params + profile.shared_params)

    def test_reusable_core_learns_r440_controls(self):
        model, factory = _train_core(seed=515, steps=1200)
        frame1, frame2, extra, target_color, operator, hard, targets = factory.make_batch()
        outputs = model(frame1, frame2, extra)

        gate_pred = (outputs.depth_gate > 0).long()
        answer_acc = _accuracy(outputs.answer, targets["answer"])
        gate_acc = float((gate_pred == targets["depth_gate"]).float().mean().item())
        fast_easy = _accuracy(outputs.fast_answer[~hard], targets["answer"][~hard])
        deep_hard = _accuracy(outputs.deep_answer[hard], targets["answer"][hard])
        depth_separation = float(outputs.depth_probability[hard].mean().item() - outputs.depth_probability[~hard].mean().item())

        no_image1, no_image2, no_image_extra, no_image_targets = factory.make_no_image_batch()
        no_image_acc = _accuracy(model(no_image1, no_image2, no_image_extra).answer, no_image_targets["answer"])
        slot_ablated_acc = _accuracy(model(frame1, frame2, extra, ablate_slots=True).answer, targets["answer"])

        swapped_extra = extra.clone()
        swapped_color = (target_color + 1 + torch.randint(0, factory.config.colors - 1, (factory.batch_size,))) % factory.config.colors
        swapped_extra[:, : factory.config.colors] = _one_hot(swapped_color, factory.config.colors)
        query_swap_acc = _accuracy(model(frame1, frame2, swapped_extra).answer, targets["answer"])

        shuffled_extra = extra.clone()
        shuffled_extra[:, factory.config.colors : factory.config.colors + factory.config.operators] = _one_hot(
            1 - operator,
            factory.config.operators,
        )
        proposition_shuffle_acc = _accuracy(model(frame1, frame2, shuffled_extra).answer, targets["answer"])
        force_fast_hard = _accuracy(model(frame1, frame2, extra, force_fast=True).answer[hard], targets["answer"][hard])

        self.assertGreaterEqual(answer_acc, 0.90)
        self.assertGreaterEqual(gate_acc, 0.90)
        self.assertGreaterEqual(fast_easy, 0.85)
        self.assertGreaterEqual(deep_hard, 0.85)
        self.assertGreaterEqual(depth_separation, 0.50)
        self.assertLessEqual(no_image_acc, 0.25)
        self.assertLessEqual(slot_ablated_acc, 0.35)
        self.assertLessEqual(query_swap_acc, 0.35)
        self.assertLessEqual(proposition_shuffle_acc, 0.35)
        self.assertLessEqual(force_fast_hard, 0.35)


if __name__ == "__main__":
    unittest.main()
