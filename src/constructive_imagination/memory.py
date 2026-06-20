"""Fixed-budget internal memory cells for R439B gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MemoryWrite:
    cell_index: int
    written: bool
    surprise: float
    conflict: float
    utility: float


class AttractorMemory:
    """Small fixed-budget memory bank with utility-gated writes."""

    def __init__(self, cells: int, width: int, *, write_threshold: float = 0.5):
        if cells <= 0:
            raise ValueError("cells must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        self.cells = np.zeros((cells, width), dtype=float)
        self.utility = np.zeros(cells, dtype=float)
        self.write_threshold = float(write_threshold)

    def write(self, vector: np.ndarray, *, surprise: float, conflict: float) -> MemoryWrite:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.cells.shape[1],):
            raise ValueError("vector width does not match memory width")
        utility = float(max(surprise, 0.0) + max(conflict, 0.0))
        index = int(np.argmin(self.utility))
        written = utility >= self.write_threshold
        if written:
            norm = np.linalg.norm(vector)
            normalized = vector / norm if norm > 0 else vector
            self.cells[index] = normalized
            self.utility[index] = utility
        return MemoryWrite(
            cell_index=index,
            written=written,
            surprise=float(surprise),
            conflict=float(conflict),
            utility=utility,
        )

    def recall(self, query: np.ndarray, *, top_k: int = 1) -> np.ndarray:
        query = np.asarray(query, dtype=float)
        if query.shape != (self.cells.shape[1],):
            raise ValueError("query width does not match memory width")
        scores = self.cells @ query
        k = max(1, min(int(top_k), self.cells.shape[0]))
        indices = np.argpartition(scores, -k)[-k:]
        weights = scores[indices]
        if np.allclose(weights, 0.0):
            return np.zeros_like(query)
        weights = np.maximum(weights, 0.0)
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        return weights @ self.cells[indices]

    def zero_cell(self, index: int) -> None:
        self.cells[index] = 0.0
        self.utility[index] = 0.0

