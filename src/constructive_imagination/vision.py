"""Synthetic visual write gates for same-caption twin tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import ConstructiveObject, ConstructiveState


@dataclass(frozen=True)
class SyntheticVisualScene:
    """Tiny visual scene whose caption intentionally loses hidden structure."""

    caption: str
    visible_object_id: str
    hidden_object_id: str
    visible_position: tuple[int, int]
    hidden_position: tuple[int, int]
    color: str = "red"
    kind: str = "cube"


def write_visual_state(scene: SyntheticVisualScene) -> ConstructiveState:
    """Write visual evidence into constructive state instead of a caption."""

    visible = ConstructiveObject(
        object_id=scene.visible_object_id,
        kind=scene.kind,
        position=scene.visible_position,
        attributes={"color": scene.color, "visible": True},
        provenance="visual",
    )
    hidden = ConstructiveObject(
        object_id=scene.hidden_object_id,
        kind=scene.kind,
        position=scene.hidden_position,
        attributes={"color": scene.color, "visible": False},
        provenance="hidden_continuation",
    )
    boundary = np.zeros((4, 4), dtype=float)
    void = np.ones((4, 4), dtype=float)
    boundary[scene.visible_position] = 1.0
    boundary[scene.hidden_position] = 0.5
    void[scene.visible_position] = 0.0
    void[scene.hidden_position] = 0.25

    return ConstructiveState(
        objects={scene.visible_object_id: visible},
        boundary=boundary,
        void=void,
        hidden={scene.hidden_object_id: hidden},
        debt={"hidden_continuation": 1.0, "visual_repair": 0.5},
        provenance={
            f"object:{scene.visible_object_id}": "visual",
            f"hidden:{scene.hidden_object_id}": "visual_occlusion",
        },
    )


def delayed_reveal_prediction(
    state: ConstructiveState,
    hidden_object_id: str,
) -> tuple[int, int] | None:
    hidden = state.hidden.get(hidden_object_id)
    if isinstance(hidden, ConstructiveObject):
        return hidden.position
    obj = state.objects.get(hidden_object_id)
    if obj is not None:
        return obj.position
    return None


def caption_only_hidden_prediction(caption: str) -> tuple[int, int]:
    """Caption-only control intentionally cannot distinguish same-caption twins."""

    _ = caption
    return (0, 0)

