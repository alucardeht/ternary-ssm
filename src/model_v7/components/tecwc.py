"""Ternary Error-Correcting Weight Codes (R4-A3-P2).

Constrains weight vectors to lie in a ternary error-correcting code [N,K,d].
Projection: nearest_codeword(W) via syndrome decoding over GF(3).
Simplified: use repetition code structure for practical ternary ECC.
"""
import torch


def build_ternary_repetition_matrix(n: int, k: int, device: torch.device) -> torch.Tensor:
    """Build a simple ternary generator matrix G (k x n) over GF(3).

    Uses systematic form: G = [I_k | P] where P adds parity checks.
    This gives distance d = 2 (can detect 1 error).

    For higher distance, would need proper BCH over GF(3).
    """
    assert n >= k, f"n={n} must be >= k={k}"
    G = torch.zeros(k, n, device=device)
    # Identity part
    G[:k, :k] = torch.eye(k, device=device)
    # Parity part: simple checksum pattern
    r = n - k
    if r > 0:
        for i in range(k):
            for j in range(r):
                G[i, k + j] = ((i + j + 1) % 3)  # GF(3) pattern
    return G


def project_to_code(w: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Project weight vector to nearest codeword.

    Simplified: quantize to GF(3) = {0, 1, 2} -> map to {-1, 0, +1},
    then find nearest codeword via syndrome.

    For practical use, this is a soft projection that biases weights
    toward code-valid configurations.

    Args:
        w: weight tensor (any shape, last dim is projected)
        G: generator matrix (k x n) over GF(3)

    Returns:
        Projected weights (same shape, values in {-1, 0, +1}).
    """
    original_shape = w.shape
    d = w.shape[-1]
    k, n = G.shape

    if d != n:
        return torch.sign(w) * (w.abs() > w.abs().mean())  # Fallback

    # Map weights to GF(3): {-1->2, 0->0, +1->1}
    w_ternary = torch.sign(w) * (w.abs() > w.abs().mean())
    w_gf3 = ((w_ternary + 1) % 3).long()  # {-1->0, 0->1, +1->2} -> map

    # Syndrome: S = w * H^T mod 3 where H is parity check
    # For systematic code, H = [-P^T | I_r]
    # Simplified: just ensure parity constraints are approximately met

    # Information bits (first k)
    info = w_gf3[..., :k]

    # Re-encode: codeword = info @ G mod 3
    G_long = G.long()
    encoded = (info.float() @ G_long.float()) % 3  # (..., n)

    # Map back: {0->0, 1->+1, 2->-1}
    result = torch.zeros_like(w)
    result[encoded == 1] = 1.0
    result[encoded == 2] = -1.0

    return result


def compute_tecwc_regularizer(
    model: torch.nn.Module,
    code_rate: float = 0.9,
    lambda_code: float = 0.001,
) -> torch.Tensor:
    """Compute ECC regularization loss.

    Penalizes distance between current ternary weights and nearest codeword.

    Args:
        model: model with TernaryLinear modules
        code_rate: R = K/N, fraction of information bits (0.9 = 10% redundancy)
        lambda_code: regularization weight

    Returns:
        Scalar regularization loss.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    n_layers = 0

    for module in model.modules():
        if not hasattr(module, 'quantize_enabled') or not hasattr(module, 'weight'):
            continue

        w = module.weight.data
        if w.dim() < 2:
            continue

        out_dim, in_dim = w.shape[0], w.shape[1]
        k = max(1, int(in_dim * code_rate))
        n = in_dim

        # Build code for this layer's dimension
        G = build_ternary_repetition_matrix(n, k, w.device)

        # Current ternary
        threshold = getattr(module, 'threshold_coeff', 0.5) * w.abs().mean()
        w_ternary = torch.sign(w) * (w.abs() > threshold).float()

        # Nearest codeword
        w_code = project_to_code(w_ternary, G)

        # Distance (Hamming)
        dist = (w_ternary != w_code).float().mean()
        total = total + dist
        n_layers += 1

    return lambda_code * total / max(n_layers, 1)
