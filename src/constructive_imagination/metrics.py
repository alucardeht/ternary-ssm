"""Cost and parameter metrics for constructive imagination models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from .core import ConstructiveCognitiveCore


@dataclass(frozen=True)
class CoreCostProfile:
    total_params: int
    shared_params: int
    fast_path_params: int
    deep_path_params: int
    gate_params: int
    expected_active_params: float
    route_probability: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def profile_constructive_core(
    model: ConstructiveCognitiveCore,
    *,
    route_probability: float,
) -> CoreCostProfile:
    shared = count_parameters(model.query_encoder) + count_parameters(model.proposition_compiler)
    fast = (
        count_parameters(model.fast_token_encoder)
        + count_parameters(model.fast_score)
        + count_parameters(model.fast_value)
        + count_parameters(model.fast_reader)
        + count_parameters(model.fast_answer)
    )
    deep = (
        count_parameters(model.deep_token_encoder)
        + count_parameters(model.deep_score)
        + count_parameters(model.deep_value)
        + count_parameters(model.deep_reader)
        + count_parameters(model.deep_answer)
    )
    gate = count_parameters(model.depth_gate)
    route = max(0.0, min(1.0, float(route_probability)))
    expected = shared + gate + fast + route * deep
    return CoreCostProfile(
        total_params=count_parameters(model),
        shared_params=shared,
        fast_path_params=fast,
        deep_path_params=deep,
        gate_params=gate,
        expected_active_params=expected,
        route_probability=route,
    )

