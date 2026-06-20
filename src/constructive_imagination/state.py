"""Constructive state and typed local patch primitives."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConstructiveObject:
    """Object carrier with identity, position, and grounded attributes."""

    object_id: str
    kind: str
    position: tuple[int, int]
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: str = "observed"


@dataclass(frozen=True)
class ConstructiveState:
    """Minimal R439B constructive state.

    Coordinates map to the R439B notation:
    - objects: `O_t`
    - relations/boundary/void fields: `phi_t` and `V_t`
    - hidden: `H_t`
    - debt: `D_t`
    """

    objects: dict[str, ConstructiveObject] = field(default_factory=dict)
    boundary: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    void: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    hidden: dict[str, Any] = field(default_factory=dict)
    debt: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def total_debt(self) -> float:
        return float(sum(max(0.0, value) for value in self.debt.values()))

    def with_debt_delta(self, key: str, delta: float) -> "ConstructiveState":
        debt = dict(self.debt)
        debt[key] = max(0.0, float(debt.get(key, 0.0) + delta))
        return replace(self, debt=debt)


@dataclass(frozen=True)
class StatePatch:
    """Typed local state edit with explicit support and provenance."""

    patch_type: str
    target: str
    payload: Any
    support: tuple[str, ...] = ()
    debt_key: str | None = None
    expected_debt_delta: float = 0.0
    provenance: str = "internal_patch"


def _copy_state(state: ConstructiveState) -> dict[str, Any]:
    return {
        "objects": dict(state.objects),
        "boundary": np.array(state.boundary, copy=True),
        "void": np.array(state.void, copy=True),
        "hidden": dict(state.hidden),
        "debt": dict(state.debt),
        "provenance": dict(state.provenance),
    }


def apply_patch(state: ConstructiveState, patch: StatePatch) -> ConstructiveState:
    """Apply a typed local patch and update repair debt.

    This is deliberately small and strict: unsupported global latent jumps fail
    instead of silently becoming free-form state mutation.
    """

    data = _copy_state(state)

    if patch.patch_type == "move_object":
        if patch.target not in data["objects"]:
            raise KeyError(f"unknown object target: {patch.target}")
        obj = data["objects"][patch.target]
        data["objects"][patch.target] = replace(obj, position=tuple(patch.payload))
        data["provenance"][f"object:{patch.target}:position"] = patch.provenance
    elif patch.patch_type == "set_attribute":
        if patch.target not in data["objects"]:
            raise KeyError(f"unknown object target: {patch.target}")
        name, value = patch.payload
        obj = data["objects"][patch.target]
        attributes = dict(obj.attributes)
        attributes[name] = value
        data["objects"][patch.target] = replace(obj, attributes=attributes)
        data["provenance"][f"object:{patch.target}:attribute:{name}"] = patch.provenance
    elif patch.patch_type == "repair_boundary":
        row, col, value = patch.payload
        data["boundary"][row, col] = value
        data["provenance"][f"boundary:{row}:{col}"] = patch.provenance
    elif patch.patch_type == "reveal_hidden":
        if patch.target not in data["hidden"]:
            raise KeyError(f"unknown hidden target: {patch.target}")
        hidden_obj = data["hidden"].pop(patch.target)
        if not isinstance(hidden_obj, ConstructiveObject):
            raise TypeError("hidden reveal payload must be a ConstructiveObject")
        data["objects"][patch.target] = hidden_obj
        data["provenance"][f"object:{patch.target}:reveal"] = patch.provenance
    else:
        raise ValueError(f"unsupported patch type: {patch.patch_type}")

    if patch.debt_key is not None:
        current = float(data["debt"].get(patch.debt_key, 0.0))
        data["debt"][patch.debt_key] = max(0.0, current + patch.expected_debt_delta)

    return ConstructiveState(**data)

