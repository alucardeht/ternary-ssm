"""Reusable R440 constructive cognition core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConstructiveCoreConfig:
    grid: int = 4
    colors: int = 3
    actions: int = 4
    operators: int = 2
    query_dim: int = 48
    proposition_dim: int = 32
    slot_dim: int = 64
    hidden_dim: int = 96

    @property
    def cells(self) -> int:
        return self.grid * self.grid

    @property
    def frame_dim(self) -> int:
        return self.colors + 2

    @property
    def extra_dim(self) -> int:
        return self.colors + self.operators + self.actions


@dataclass
class ConstructiveCoreOutput:
    answer: torch.Tensor
    fast_answer: torch.Tensor
    deep_answer: torch.Tensor
    depth_gate: torch.Tensor
    depth_probability: torch.Tensor
    fast_attention: torch.Tensor
    deep_attention: torch.Tensor
    route_rate: torch.Tensor


class ConstructiveCognitiveCore(nn.Module):
    """Small trainable core for query-keyed slots, propositions, and depth."""

    def __init__(self, config: ConstructiveCoreConfig | None = None):
        super().__init__()
        self.config = config or ConstructiveCoreConfig()
        cfg = self.config

        self.query_encoder = nn.Sequential(nn.Linear(cfg.extra_dim, cfg.query_dim), nn.ReLU())
        self.proposition_compiler = nn.Sequential(
            nn.Linear(cfg.extra_dim, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.proposition_dim),
            nn.Tanh(),
        )
        self.fast_token_encoder = nn.Sequential(
            nn.Linear(cfg.frame_dim + cfg.query_dim + cfg.proposition_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.deep_token_encoder = nn.Sequential(
            nn.Linear(cfg.frame_dim + 2 + cfg.query_dim + cfg.proposition_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fast_score = nn.Linear(cfg.hidden_dim, 1)
        self.deep_score = nn.Linear(cfg.hidden_dim, 1)
        self.fast_value = nn.Linear(cfg.hidden_dim, cfg.slot_dim)
        self.deep_value = nn.Linear(cfg.hidden_dim, cfg.slot_dim)
        self.fast_reader = nn.Sequential(
            nn.Linear(cfg.slot_dim + cfg.proposition_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.deep_reader = nn.Sequential(
            nn.Linear(cfg.slot_dim + cfg.proposition_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fast_answer = nn.Linear(cfg.hidden_dim, cfg.cells)
        self.deep_answer = nn.Linear(cfg.hidden_dim, cfg.cells)
        self.depth_gate = nn.Linear(cfg.hidden_dim + cfg.proposition_dim + 1, 1)

    def _compile(self, extra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_encoder(extra), self.proposition_compiler(extra)

    def _select(
        self,
        encoded: torch.Tensor,
        score: nn.Linear,
        value: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.softmax(score(encoded).squeeze(-1), dim=-1)
        slot = torch.sum(attention.unsqueeze(-1) * value(encoded), dim=1)
        return slot, attention

    def forward_fast(self, frame2: torch.Tensor, extra: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        query, proposition = self._compile(extra)
        context = torch.cat(
            (
                frame2,
                query.unsqueeze(1).expand(-1, cfg.cells, -1),
                proposition.unsqueeze(1).expand(-1, cfg.cells, -1),
            ),
            dim=-1,
        )
        slot, attention = self._select(self.fast_token_encoder(context), self.fast_score, self.fast_value)
        h = self.fast_reader(torch.cat((slot, proposition), dim=-1))
        return {
            "logits": self.fast_answer(h),
            "hidden": h,
            "attention": attention,
            "proposition": proposition,
        }

    def forward_deep(self, frame1: torch.Tensor, extra: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        query, proposition = self._compile(extra)
        frame_tag = frame1.new_tensor([1.0, 0.0]).view(1, 1, 2).expand(frame1.shape[0], cfg.cells, 2)
        context = torch.cat(
            (
                torch.cat((frame1, frame_tag), dim=-1),
                query.unsqueeze(1).expand(-1, cfg.cells, -1),
                proposition.unsqueeze(1).expand(-1, cfg.cells, -1),
            ),
            dim=-1,
        )
        slot, attention = self._select(self.deep_token_encoder(context), self.deep_score, self.deep_value)
        h = self.deep_reader(torch.cat((slot, proposition), dim=-1))
        return {
            "logits": self.deep_answer(h),
            "hidden": h,
            "attention": attention,
            "proposition": proposition,
        }

    def route(self, frame2: torch.Tensor, extra: torch.Tensor, fast_hidden: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        proposition = self.proposition_compiler(extra)
        target_presence = torch.sum(
            frame2[:, :, : cfg.colors] * extra[:, : cfg.colors].unsqueeze(1),
            dim=(1, 2),
        ).unsqueeze(-1)
        return self.depth_gate(torch.cat((fast_hidden, proposition, target_presence), dim=-1)).squeeze(-1)

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        extra: torch.Tensor,
        *,
        ablate_slots: bool = False,
        force_fast: bool = False,
        force_deep: bool = False,
    ) -> ConstructiveCoreOutput:
        cfg = self.config
        fast = self.forward_fast(frame2, extra)
        deep = self.forward_deep(frame1, extra)
        fast_logits = fast["logits"]
        deep_logits = deep["logits"]
        fast_attention = fast["attention"]
        deep_attention = deep["attention"]

        if ablate_slots:
            fast_logits = torch.zeros_like(fast_logits)
            deep_logits = torch.zeros_like(deep_logits)
            fast_attention = torch.full_like(fast_attention, 1.0 / cfg.cells)
            deep_attention = torch.full_like(deep_attention, 1.0 / cfg.cells)

        gate_logit = self.route(frame2, extra, fast["hidden"])
        gate = torch.sigmoid(gate_logit).unsqueeze(-1)
        if force_fast:
            gate = torch.zeros_like(gate)
        if force_deep:
            gate = torch.ones_like(gate)
        return ConstructiveCoreOutput(
            answer=fast_logits * (1.0 - gate) + deep_logits * gate,
            fast_answer=fast_logits,
            deep_answer=deep_logits,
            depth_gate=gate_logit,
            depth_probability=gate.squeeze(-1),
            fast_attention=fast_attention,
            deep_attention=deep_attention,
            route_rate=gate.mean(),
        )

