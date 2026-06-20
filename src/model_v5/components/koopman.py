"""Koopman Spectral Shaping (R4-A3-P7).

RecursiveSSM as dynamical system. Shape Koopman eigenvalues toward target spectrum.
EDMD: K_approx = Phi_Y @ pinv(Phi_X)
"""
import torch


def compute_koopman_operator(
    trajectory: list,
    max_dim: int = 64,
) -> torch.Tensor:
    """Estimate Koopman operator from trajectory via EDMD.

    Args:
        trajectory: list of (batch, seq, d_model) states
        max_dim: truncate to this dimension for efficiency

    Returns:
        K: (d, d) estimated Koopman operator.
    """
    if len(trajectory) < 2:
        return None

    d = min(trajectory[0].shape[-1], max_dim)
    device = trajectory[0].device

    # X = [h_0, ..., h_{L-2}], Y = [h_1, ..., h_{L-1}]
    X_list = []
    Y_list = []

    for i in range(len(trajectory) - 1):
        # Average over batch and seq for a single state vector
        x = trajectory[i].detach().mean(dim=(0, 1))[:d]
        y = trajectory[i + 1].detach().mean(dim=(0, 1))[:d]
        X_list.append(x)
        Y_list.append(y)

    X = torch.stack(X_list)  # (L-1, d)
    Y = torch.stack(Y_list)  # (L-1, d)

    # EDMD: K = Y^T @ X @ (X^T @ X)^{-1}  (least squares)
    XtX = X.T @ X + 1e-6 * torch.eye(d, device=device)  # Regularize
    K = Y.T @ X @ torch.linalg.inv(XtX)

    return K


def compute_koopman_spectral_loss(
    K: torch.Tensor,
    target_long_memory: float = 0.97,
    target_fast: float = 0.5,
    long_fraction: float = 0.3,
) -> torch.Tensor:
    """Penalize deviation of Koopman eigenvalues from target spectrum.

    Args:
        K: (d, d) Koopman operator
        target_long_memory: target |eigenvalue| for long-memory modes
        target_fast: target |eigenvalue| for fast-processing modes
        long_fraction: fraction of modes that should be long-memory

    Returns:
        Scalar spectral shaping loss.
    """
    if K is None:
        return torch.tensor(0.0)

    d = K.shape[0]

    try:
        eigvals = torch.linalg.eigvals(K)
        mags = eigvals.abs()
        mags_sorted, _ = mags.sort(descending=True)
    except Exception:
        return torch.tensor(0.0, device=K.device)

    n_long = max(1, int(d * long_fraction))
    n_fast = d - n_long

    loss = torch.tensor(0.0, device=K.device)

    # Long-memory modes should have |lambda| near target
    if n_long > 0:
        long_mags = mags_sorted[:n_long]
        loss = loss + ((long_mags - target_long_memory) ** 2).mean()

    # Fast modes should have |lambda| near target_fast
    if n_fast > 0:
        fast_mags = mags_sorted[n_long:]
        loss = loss + ((fast_mags - target_fast) ** 2).mean()

    return loss
