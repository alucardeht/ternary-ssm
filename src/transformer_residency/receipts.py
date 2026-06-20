"""Byte receipts for transformer residency tile candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileReceipt:
    """Cost receipt for one executable tile representation."""

    dense_elements: int
    payload_bits_per_element: float
    metadata_bytes: int = 0
    scale_bytes: int = 0
    correction_bytes: int = 0
    scratch_bytes: int = 0
    launch_fragmentation_bytes: int = 0
    dense_bytes_per_element: int = 2

    @property
    def dense_bytes(self) -> float:
        return float(self.dense_elements * self.dense_bytes_per_element)

    @property
    def payload_bytes(self) -> float:
        return float(self.dense_elements * self.payload_bits_per_element / 8.0)

    @property
    def execution_bytes(self) -> float:
        return (
            self.payload_bytes
            + self.metadata_bytes
            + self.scale_bytes
            + self.correction_bytes
            + self.scratch_bytes
            + self.launch_fragmentation_bytes
        )

    @property
    def saved_bytes(self) -> float:
        return self.dense_bytes - self.execution_bytes

    @property
    def effective_bits_per_element(self) -> float:
        if self.dense_elements <= 0:
            return 0.0
        return 8.0 * self.execution_bytes / self.dense_elements

    @property
    def compression_ratio_vs_fp16(self) -> float:
        if self.execution_bytes <= 0:
            return float("inf")
        return self.dense_bytes / self.execution_bytes

    def passes(self, min_saved_fraction: float = 0.10) -> bool:
        if self.dense_bytes <= 0:
            return False
        return self.saved_bytes >= self.dense_bytes * min_saved_fraction

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "dense_elements": self.dense_elements,
            "dense_bytes": self.dense_bytes,
            "payload_bits_per_element": self.payload_bits_per_element,
            "payload_bytes": self.payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "scale_bytes": self.scale_bytes,
            "correction_bytes": self.correction_bytes,
            "scratch_bytes": self.scratch_bytes,
            "launch_fragmentation_bytes": self.launch_fragmentation_bytes,
            "execution_bytes": self.execution_bytes,
            "saved_bytes": self.saved_bytes,
            "effective_bits_per_element": self.effective_bits_per_element,
            "compression_ratio_vs_fp16": self.compression_ratio_vs_fp16,
            "passes_10pct_saved_gate": self.passes(0.10),
        }
