"""Spectral radius utilities for SSM analysis."""
import torch
import math


def compute_spectral_radius(A_log: torch.Tensor) -> float:
    """Compute spectral radius of SSM A matrix.

    For diagonal SSM (Mamba-style), A = -exp(A_log), so eigenvalues are -exp(A_log).
    The spectral radius is max(|eigenvalue|) = max(exp(A_log)).

    For discrete-time (after ZOH): A_d = exp(A * dt).
    With diagonal A, spectral radius of A_d = max(exp(A_log * dt)).

    Args:
        A_log: Log-parameterized A matrix [d_state] or [d_model, d_state]

    Returns:
        Spectral radius (float). Values < 1 mean state decays, > 1 means unstable.
    """
    with torch.no_grad():
        # A = -exp(A_log), eigenvalues are -exp(A_log)
        # |eigenvalue| = exp(A_log)
        # spectral_radius = max(exp(A_log))
        return torch.exp(A_log).max().item()


def compute_safe_sequence_length(spectral_radius: float, tolerance: float = 0.01) -> int:
    """Compute the sequence length at which SSM state decays below tolerance.

    If spectral_radius^L < tolerance, then after L steps the state contribution
    from before is negligible. This means we DON'T need to reset state within
    sequences shorter than L.

    Args:
        spectral_radius: Max eigenvalue magnitude of discrete A
        tolerance: State decay threshold (default 1% of original magnitude)

    Returns:
        Safe sequence length L. Document boundaries within L tokens don't need resets.
    """
    if spectral_radius <= 0 or spectral_radius >= 1:
        return 1  # Can't determine safely, always reset

    # spectral_radius^L < tolerance
    # L > log(tolerance) / log(spectral_radius)
    L = math.ceil(math.log(tolerance) / math.log(spectral_radius))
    return max(1, L)
