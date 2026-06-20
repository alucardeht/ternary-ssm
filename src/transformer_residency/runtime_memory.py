"""Process memory snapshots for transformer residency runs."""

from __future__ import annotations

import os
import sys
from typing import Any

from .footprint import BYTES_PER_GIB


def process_memory_snapshot() -> dict[str, Any]:
    """Return a best-effort process RAM snapshot in GiB.

    On Windows this uses the native process counters so each benchmark process
    reports its own peak working set and current private usage without relying
    on a third-party dependency. Non-Windows falls back to ``resource`` where
    available.
    """

    if sys.platform.startswith("win"):
        return _windows_process_memory_snapshot()
    return _resource_process_memory_snapshot()


def _windows_process_memory_snapshot() -> dict[str, Any]:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return {
            "source": "windows_psapi",
            "peak_working_set_gib": None,
            "working_set_gib": None,
            "private_usage_gib": None,
        }
    return {
        "source": "windows_psapi",
        "peak_working_set_gib": counters.PeakWorkingSetSize / BYTES_PER_GIB,
        "working_set_gib": counters.WorkingSetSize / BYTES_PER_GIB,
        "private_usage_gib": counters.PrivateUsage / BYTES_PER_GIB,
    }


def _resource_process_memory_snapshot() -> dict[str, Any]:
    try:
        import resource
    except Exception:
        return {
            "source": "unavailable",
            "peak_working_set_gib": None,
            "working_set_gib": None,
            "private_usage_gib": None,
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_bytes = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        peak_bytes *= 1.0
    else:
        peak_bytes *= 1024.0
    return {
        "source": "resource",
        "peak_working_set_gib": peak_bytes / BYTES_PER_GIB,
        "working_set_gib": None,
        "private_usage_gib": None,
        "pid": os.getpid(),
    }
