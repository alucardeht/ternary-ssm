"""
Optimizer State Recycling Pool (R4-A6-P2, R4-A6-M7)

Memory pool that reuses freed optimizer state buffers when weights transition between
commitment states. Committed weights release their Adam state (m, v); uncommitting
weights immediately grab a buffer from the pool instead of allocating new memory.

Dramatically reduces fragmentation and allocation latency for commit/uncommit cycles.
Pool size scales with number of active (uncommitted) parameters + margin.
"""

from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import torch


class RecyclingPool:
    """
    Fixed-size pool of pre-allocated optimizer state buffers.

    When a weight commits (no longer needs Adam state), its m,v buffers are donated
    to the pool. When a weight uncommits, it acquires a buffer instantly from the pool,
    avoiding malloc overhead. If pool exhausted, forces commitment of least-confident
    layer to free space.

    Pool size = expected active layers + margin (typically 2).
    """

    def __init__(
        self,
        pool_max_mb: int = 256,
        margin_buffers: int = 2,
    ):
        """
        Args:
            pool_max_mb: Maximum pool size in MB (auto-scales down if not enough VRAM)
            margin_buffers: Extra buffers beyond expected active count (for safety margin)
        """
        self.max_size_bytes = pool_max_mb * (1024 ** 2)
        self.margin = margin_buffers

        self.buffer_pool: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque()
        self.current_size_bytes = 0

        self.allocation_history: List[float] = []
        self.hit_count = 0
        self.miss_count = 0

        self.active_buffers: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def donate(
        self,
        m_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
    ) -> bool:
        """
        Donate freed optimizer state buffers to the pool.

        Returns True if donation accepted, False if pool full.
        """
        m_bytes = m_buffer.numel() * m_buffer.element_size()
        v_bytes = v_buffer.numel() * v_buffer.element_size()
        total_bytes = m_bytes + v_bytes

        if self.current_size_bytes + total_bytes > self.max_size_bytes:
            return False

        m_cpu = m_buffer.cpu().detach()
        v_cpu = v_buffer.cpu().detach()

        self.buffer_pool.append((m_cpu, v_cpu))
        self.current_size_bytes += total_bytes
        self.allocation_history.append(float(total_bytes) / (1024 ** 2))

        return True

    def request(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Request a buffer pair (m, v) from the pool.

        Attempts to find a compatible size in the pool. If found, returns it;
        otherwise returns None (caller should allocate fresh).

        Returns:
            Tuple of (m_buffer, v_buffer) if hit, None if miss
        """
        if not self.buffer_pool:
            self.miss_count += 1
            return None

        m_buf, v_buf = self.buffer_pool.popleft()
        target_numel = 1
        for s in shape:
            target_numel *= s

        if m_buf.numel() != target_numel or v_buf.numel() != target_numel:
            self.miss_count += 1
            return None

        m_buf = m_buf.to(device=device, dtype=dtype)
        v_buf = v_buf.to(device=device, dtype=dtype)

        self.current_size_bytes -= (m_buf.numel() * m_buf.element_size() +
                                     v_buf.numel() * v_buf.element_size())

        self.hit_count += 1
        return (m_buf, v_buf)

    def acquire(
        self,
        param_id: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Acquire a buffer pair, either from pool or freshly allocated.

        Tracks which param owns the buffer via param_id.

        Args:
            param_id: Unique identifier for the parameter
            shape: Shape of the parameter (used to size m, v)
            dtype: Data type (usually fp32 for Adam state)
            device: Device placement (cpu or cuda)

        Returns:
            Tuple (m_buffer, v_buffer) ready for Adam updates
        """
        pooled = self.request(shape, dtype, device)

        if pooled is not None:
            m, v = pooled
            m.zero_()
            v.zero_()
        else:
            m = torch.zeros(shape, dtype=dtype, device=device)
            v = torch.zeros(shape, dtype=dtype, device=device)

        self.active_buffers[param_id] = (m, v)
        return (m, v)

    def release(self, param_id: str) -> bool:
        """
        Release a buffer pair when a parameter becomes committed.

        Returns True if successfully donated to pool, False if pool full.
        """
        if param_id not in self.active_buffers:
            return False

        m, v = self.active_buffers.pop(param_id)
        return self.donate(m, v)

    def get_pool_stats(self) -> Dict[str, float]:
        """
        Return comprehensive pool utilization statistics.

        Returns:
            Dict with keys: pool_size_mb, hit_rate, miss_count, active_buffers, etc.
        """
        total_attempts = max(1, self.hit_count + self.miss_count)
        hit_rate = self.hit_count / total_attempts

        avg_buffer_size = (
            sum(self.allocation_history) / max(1, len(self.allocation_history))
        )

        return {
            "pool_size_mb": self.current_size_bytes / (1024 ** 2),
            "pool_max_mb": self.max_size_bytes / (1024 ** 2),
            "pool_utilization_pct": 100.0 * self.current_size_bytes / self.max_size_bytes,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "active_buffers": len(self.active_buffers),
            "avg_buffer_size_mb": avg_buffer_size,
            "total_buffers_in_pool": len(self.buffer_pool),
        }

    def clear_pool(self) -> None:
        """Clear all pooled buffers (e.g., at end of training)."""
        self.buffer_pool.clear()
        self.current_size_bytes = 0
        self.active_buffers.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_active_buffer_ids(self) -> List[str]:
        """Return list of currently active (unreleased) buffer IDs."""
        return list(self.active_buffers.keys())
