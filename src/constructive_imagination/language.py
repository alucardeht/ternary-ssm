"""Language interface heads over constructive state."""

from __future__ import annotations

from dataclasses import dataclass

from .state import ConstructiveState


@dataclass(frozen=True)
class LanguageAnswer:
    text: str
    provenance: dict[str, str]
    abstained: bool = False


def count_red_cubes(state: ConstructiveState) -> LanguageAnswer:
    red_cubes = [
        obj
        for obj in state.objects.values()
        if obj.kind == "cube" and obj.attributes.get("color") == "red"
    ]
    if not red_cubes:
        return LanguageAnswer(
            text="unknown",
            provenance={"unknown": "missing:O_t.red_cube"},
            abstained=True,
        )
    provenance = {
        str(len(red_cubes)): "O_t.objects",
        "red": "O_t.attributes.color",
        "cubes": "O_t.objects.kind",
    }
    return LanguageAnswer(text=f"{len(red_cubes)} red cube(s)", provenance=provenance)


def canonical_query_operator(text: str) -> str:
    normalized = " ".join(text.casefold().replace("à", "a").split())
    left_forms = (
        "left of",
        "to the left of",
        "a esquerda de",
        "esquerda do",
        "esquerda da",
        "izquierda de",
    )
    count_forms = ("how many", "quantos", "quantas", "cuantos", "cuantas")
    if any(form in normalized for form in left_forms):
        return "relation:left_of"
    if any(form in normalized for form in count_forms):
        return "query:count"
    return f"literal:{normalized}"


def bind_handle(
    state: ConstructiveState,
    handle: str,
    object_id: str,
    *,
    event: str,
) -> ConstructiveState:
    if object_id not in state.objects:
        raise KeyError(f"unknown object target: {object_id}")
    hidden = dict(state.hidden)
    hidden[f"handle:{handle}"] = {
        "object_id": object_id,
        "event": event,
    }
    provenance = dict(state.provenance)
    provenance[f"handle:{handle}"] = f"H_t.{event}->{object_id}"
    return ConstructiveState(
        objects=dict(state.objects),
        boundary=state.boundary.copy(),
        void=state.void.copy(),
        hidden=hidden,
        debt=dict(state.debt),
        provenance=provenance,
    )


def resolve_handle(state: ConstructiveState, handle: str) -> str | None:
    record = state.hidden.get(f"handle:{handle}")
    if isinstance(record, dict):
        object_id = record.get("object_id")
        return str(object_id) if object_id in state.objects else None
    return None

