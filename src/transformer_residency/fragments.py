"""Fragment-clause feasibility checks for executable residency contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FragmentClause:
    name: str
    saved_bytes: int
    metadata_bytes: int
    correction_bytes: int
    scratch_bytes: int
    register_pressure: int
    max_register_pressure: int
    requires_second_pass: bool = False
    requires_dense_reconstruction: bool = False

    @property
    def hidden_bytes(self) -> int:
        return self.metadata_bytes + self.correction_bytes + self.scratch_bytes

    @property
    def net_saved_bytes(self) -> int:
        return self.saved_bytes - self.hidden_bytes

    @property
    def passes(self) -> bool:
        return (
            self.net_saved_bytes > 0
            and self.register_pressure <= self.max_register_pressure
            and not self.requires_second_pass
            and not self.requires_dense_reconstruction
        )

    def as_dict(self) -> dict[str, int | str | bool]:
        data = asdict(self)
        data["hidden_bytes"] = self.hidden_bytes
        data["net_saved_bytes"] = self.net_saved_bytes
        data["passes"] = self.passes
        return data


@dataclass(frozen=True)
class FragmentScheduleReport:
    clauses: tuple[FragmentClause, ...]

    @property
    def total_saved_bytes(self) -> int:
        return sum(clause.saved_bytes for clause in self.clauses)

    @property
    def total_hidden_bytes(self) -> int:
        return sum(clause.hidden_bytes for clause in self.clauses)

    @property
    def net_saved_bytes(self) -> int:
        return self.total_saved_bytes - self.total_hidden_bytes

    @property
    def passes(self) -> bool:
        return self.net_saved_bytes > 0 and all(clause.passes for clause in self.clauses)

    def as_dict(self) -> dict[str, object]:
        return {
            "passes": self.passes,
            "total_saved_bytes": self.total_saved_bytes,
            "total_hidden_bytes": self.total_hidden_bytes,
            "net_saved_bytes": self.net_saved_bytes,
            "clauses": [clause.as_dict() for clause in self.clauses],
        }


def compile_fragment_schedule(clauses: list[FragmentClause] | tuple[FragmentClause, ...]) -> FragmentScheduleReport:
    return FragmentScheduleReport(tuple(clauses))
