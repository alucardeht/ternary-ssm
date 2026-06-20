"""Tiny verbalizer heads for R440 latent answers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .core import ConstructiveCognitiveCore, ConstructiveCoreConfig
from .curriculum import R440Curriculum, one_hot

R440_TINY_VOCAB: tuple[str, ...] = (
    "<pad>",
    "cell",
    "at",
    "row",
    "col",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
)
NUMBER_OFFSET = R440_TINY_VOCAB.index("zero")
R440_TINY_VERBALIZER_TEMPLATES: tuple[str, ...] = ("cell", "row_col")
R440_TINY_VERBALIZER_LENGTH = 5


@dataclass
class R440VerbalizerOutput:
    token_logits: torch.Tensor


@dataclass(frozen=True)
class R440VerbalizerResult:
    seed: int
    steps: int
    passed: bool
    metrics: dict[str, float]
    examples: list[dict[str, str]]


class R440TinyVerbalizer(nn.Module):
    def __init__(self, config: ConstructiveCoreConfig | None = None):
        super().__init__()
        self.config = config or ConstructiveCoreConfig()
        self.template_count = len(R440_TINY_VERBALIZER_TEMPLATES)
        hidden = 64
        self.net = nn.Sequential(
            nn.Linear(self.config.cells + self.template_count, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, R440_TINY_VERBALIZER_LENGTH * len(R440_TINY_VOCAB)),
        )

    def forward(self, answer_logits: torch.Tensor, template_id: torch.Tensor) -> R440VerbalizerOutput:
        answer_distribution = torch.softmax(answer_logits, dim=-1)
        template = one_hot(template_id, self.template_count).to(answer_logits.device)
        logits = self.net(torch.cat((answer_distribution, template), dim=-1))
        return R440VerbalizerOutput(
            token_logits=logits.view(answer_logits.shape[0], R440_TINY_VERBALIZER_LENGTH, len(R440_TINY_VOCAB))
        )


def r440_verbalizer_targets(answer: torch.Tensor, template_id: torch.Tensor, *, grid: int = 4) -> torch.Tensor:
    target = torch.zeros(answer.shape[0], R440_TINY_VERBALIZER_LENGTH, dtype=torch.long, device=answer.device)
    row = answer // grid
    col = answer % grid
    cell_template = template_id == 0
    row_col_template = ~cell_template
    target[cell_template, 0] = R440_TINY_VOCAB.index("cell")
    target[cell_template, 1] = NUMBER_OFFSET + answer[cell_template]
    target[row_col_template, 0] = R440_TINY_VOCAB.index("at")
    target[row_col_template, 1] = R440_TINY_VOCAB.index("row")
    target[row_col_template, 2] = NUMBER_OFFSET + row[row_col_template]
    target[row_col_template, 3] = R440_TINY_VOCAB.index("col")
    target[row_col_template, 4] = NUMBER_OFFSET + col[row_col_template]
    return target


def r440_verbalizer_loss(output: R440VerbalizerOutput, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(output.token_logits.view(-1, len(R440_TINY_VOCAB)), target.view(-1))


def r440_sequence_accuracy(token_logits: torch.Tensor, target: torch.Tensor) -> float:
    predicted = token_logits.argmax(dim=-1)
    return float((predicted == target).all(dim=-1).float().mean().item())


def r440_decode_tokens(tokens: torch.Tensor) -> str:
    words = [R440_TINY_VOCAB[int(token)] for token in tokens if int(token) != 0]
    return " ".join(words)


def r440_tokens_to_cell(tokens: torch.Tensor, *, grid: int = 4) -> torch.Tensor:
    predicted = torch.full((tokens.shape[0],), -1, dtype=torch.long, device=tokens.device)
    cell_form = tokens[:, 0] == R440_TINY_VOCAB.index("cell")
    predicted[cell_form] = tokens[cell_form, 1] - NUMBER_OFFSET
    row_col_form = tokens[:, 0] == R440_TINY_VOCAB.index("at")
    row = tokens[row_col_form, 2] - NUMBER_OFFSET
    col = tokens[row_col_form, 4] - NUMBER_OFFSET
    predicted[row_col_form] = row * grid + col
    valid = (predicted >= 0) & (predicted < grid * grid)
    return torch.where(valid, predicted, torch.full_like(predicted, -1))


def evaluate_r440_verbalizer(
    model: ConstructiveCognitiveCore,
    verbalizer: R440TinyVerbalizer,
    curriculum: R440Curriculum,
) -> dict[str, float]:
    batch = curriculum.make_batch("mixed")
    with torch.no_grad():
        outputs = model(batch.frame1, batch.frame2, batch.extra)
        template0 = torch.zeros(curriculum.batch_size, dtype=torch.long)
        template1 = torch.ones(curriculum.batch_size, dtype=torch.long)
        target0 = r440_verbalizer_targets(batch.targets["answer"], template0, grid=curriculum.config.grid)
        target1 = r440_verbalizer_targets(batch.targets["answer"], template1, grid=curriculum.config.grid)
        verbal0 = verbalizer(outputs.answer, template0).token_logits
        verbal1 = verbalizer(outputs.answer, template1).token_logits
        pred0 = verbal0.argmax(dim=-1)
        pred1 = verbal1.argmax(dim=-1)

        no_image = curriculum.make_no_image_batch("mixed")
        no_image_output = verbalizer(
            model(no_image.frame1, no_image.frame2, no_image.extra).answer,
            template0,
        ).token_logits
        slot_output = verbalizer(
            model(batch.frame1, batch.frame2, batch.extra, ablate_slots=True).answer,
            template0,
        ).token_logits

        swapped_extra = batch.extra.clone()
        swapped_color = (
            batch.target_color + 1 + torch.randint(0, curriculum.config.colors - 1, (curriculum.batch_size,))
        ) % curriculum.config.colors
        swapped_extra[:, : curriculum.config.colors] = one_hot(swapped_color, curriculum.config.colors)
        query_swap_output = verbalizer(model(batch.frame1, batch.frame2, swapped_extra).answer, template0).token_logits

        shuffled_extra = batch.extra.clone()
        operator_start = curriculum.config.colors
        operator_end = operator_start + curriculum.config.operators
        shuffled_extra[:, operator_start:operator_end] = one_hot(1 - batch.operator, curriculum.config.operators)
        proposition_shuffle_output = verbalizer(
            model(batch.frame1, batch.frame2, shuffled_extra).answer,
            template0,
        ).token_logits

    template0_accuracy = r440_sequence_accuracy(verbal0, target0)
    template1_accuracy = r440_sequence_accuracy(verbal1, target1)
    return {
        "template0_text": template0_accuracy,
        "template1_text": template1_accuracy,
        "text": min(template0_accuracy, template1_accuracy),
        "template_cell_agreement": float((r440_tokens_to_cell(pred0) == r440_tokens_to_cell(pred1)).float().mean().item()),
        "no_image_text": r440_sequence_accuracy(no_image_output, target0),
        "slot_ablation_text": r440_sequence_accuracy(slot_output, target0),
        "query_swap_text": r440_sequence_accuracy(query_swap_output, target0),
        "proposition_shuffle_text": r440_sequence_accuracy(proposition_shuffle_output, target0),
    }


def r440_verbalizer_passed(metrics: dict[str, float]) -> bool:
    return (
        metrics["text"] >= 0.90
        and metrics["template_cell_agreement"] >= 0.90
        and metrics["no_image_text"] <= 0.35
        and metrics["slot_ablation_text"] <= 0.35
        and metrics["query_swap_text"] <= 0.35
        and metrics["proposition_shuffle_text"] <= 0.35
    )


def train_r440_tiny_verbalizer(
    model: ConstructiveCognitiveCore,
    curriculum: R440Curriculum,
    *,
    seed: int = 101,
    steps: int = 160,
    lr: float = 0.01,
) -> tuple[R440TinyVerbalizer, R440VerbalizerResult]:
    torch.manual_seed(seed + 2000)
    verbalizer = R440TinyVerbalizer(curriculum.config)
    optimizer = torch.optim.AdamW(verbalizer.parameters(), lr=lr)
    model.eval()
    verbalizer.train()
    for _ in range(steps):
        batch = curriculum.make_batch("mixed")
        template_id = torch.randint(0, len(R440_TINY_VERBALIZER_TEMPLATES), (curriculum.batch_size,))
        with torch.no_grad():
            outputs = model(batch.frame1, batch.frame2, batch.extra)
        target = r440_verbalizer_targets(batch.targets["answer"], template_id, grid=curriculum.config.grid)
        loss = r440_verbalizer_loss(verbalizer(outputs.answer.detach(), template_id), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    verbalizer.eval()
    metrics = evaluate_r440_verbalizer(model, verbalizer, R440Curriculum(seed=seed + 3000, batch_size=curriculum.batch_size))
    examples = r440_verbalizer_examples(model, verbalizer, R440Curriculum(seed=seed + 4000, batch_size=4))
    result = R440VerbalizerResult(
        seed=seed,
        steps=steps,
        passed=r440_verbalizer_passed(metrics),
        metrics=metrics,
        examples=examples,
    )
    return verbalizer, result


def r440_verbalizer_examples(
    model: ConstructiveCognitiveCore,
    verbalizer: R440TinyVerbalizer,
    curriculum: R440Curriculum,
) -> list[dict[str, str]]:
    batch = curriculum.make_batch("mixed")
    with torch.no_grad():
        outputs = model(batch.frame1, batch.frame2, batch.extra)
        template0 = torch.zeros(curriculum.batch_size, dtype=torch.long)
        template1 = torch.ones(curriculum.batch_size, dtype=torch.long)
        pred0 = verbalizer(outputs.answer, template0).token_logits.argmax(dim=-1)
        pred1 = verbalizer(outputs.answer, template1).token_logits.argmax(dim=-1)
    examples = []
    for idx in range(curriculum.batch_size):
        expected0 = r440_verbalizer_targets(batch.targets["answer"][idx : idx + 1], template0[idx : idx + 1], grid=curriculum.config.grid)
        expected1 = r440_verbalizer_targets(batch.targets["answer"][idx : idx + 1], template1[idx : idx + 1], grid=curriculum.config.grid)
        examples.append(
            {
                "expected_cell": r440_decode_tokens(expected0[0]),
                "predicted_cell": r440_decode_tokens(pred0[idx]),
                "expected_row_col": r440_decode_tokens(expected1[0]),
                "predicted_row_col": r440_decode_tokens(pred1[idx]),
            }
        )
    return examples
