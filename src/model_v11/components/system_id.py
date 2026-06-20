"""ACSI - Analytical Construction via System Identification (R2-A3-P1).

Uses the Ho-Kalman algorithm to initialize SSM parameters from input data statistics.
For LTI layers: initializes A_log, B_fixed, C_fixed from optimal realization.
For selective layers: initializes A_log from dominant temporal mode of input data.
"""

import math
import torch
import torch.nn.functional as F


def compute_autocorrelation(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Compute per-head auto-correlation from SSM input tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim) - SSM input after conv1d + activation
        max_lag: Number of lag values to compute (typically 2 * d_state)

    Returns:
        R: (n_heads, max_lag) - auto-correlation R_h(k) for each head
    """
    batch, seq_len, n_heads, head_dim = x.shape
    # Average over batch and head_dim to get per-head scalar autocorrelation
    # x_flat: (batch * head_dim, n_heads, seq_len)
    x_flat = x.permute(0, 3, 2, 1).reshape(-1, n_heads, seq_len)  # (B*D, H, T)

    R = torch.zeros(n_heads, max_lag, device=x.device, dtype=torch.float32)
    N = x_flat.shape[0]  # batch * head_dim

    for k in range(max_lag):
        if k == 0:
            R[:, k] = (x_flat * x_flat).sum(dim=(0, 2)) / (N * seq_len)
        else:
            valid_len = seq_len - k
            if valid_len <= 0:
                break
            R[:, k] = (x_flat[:, :, :valid_len] * x_flat[:, :, k:k+valid_len]).sum(dim=(0, 2)) / (N * valid_len)

    return R


def build_hankel(R: torch.Tensor, d_state: int) -> tuple:
    """Build Hankel and shifted Hankel matrices from auto-correlation.

    Args:
        R: (n_heads, 2*d_state) - auto-correlation sequence
        d_state: SSM state dimension

    Returns:
        H: (n_heads, d_state, d_state) - Hankel matrix
        H_shift: (n_heads, d_state, d_state) - shifted Hankel matrix
    """
    n_heads = R.shape[0]
    H = torch.zeros(n_heads, d_state, d_state, device=R.device, dtype=R.dtype)
    H_shift = torch.zeros_like(H)

    for i in range(d_state):
        for j in range(d_state):
            idx = i + j
            idx_shift = i + j + 1
            if idx < R.shape[1]:
                H[:, i, j] = R[:, idx]
            if idx_shift < R.shape[1]:
                H_shift[:, i, j] = R[:, idx_shift]

    return H, H_shift


def ho_kalman_realization(H: torch.Tensor, H_shift: torch.Tensor, d_state: int) -> tuple:
    """Ho-Kalman algorithm: extract (A, B, C) from Hankel matrices via SVD.

    Args:
        H: (n_heads, d_state, d_state) - Hankel matrix
        H_shift: (n_heads, d_state, d_state) - shifted Hankel matrix
        d_state: SSM state dimension

    Returns:
        A_scalars: (n_heads,) - dominant eigenvalue per head (for scalar A parameterization)
        B_vecs: (n_heads, d_state) - B vector per head
        C_vecs: (n_heads, d_state) - C vector per head
    """
    n_heads = H.shape[0]
    A_scalars = torch.zeros(n_heads, device=H.device, dtype=H.dtype)
    B_vecs = torch.zeros(n_heads, d_state, device=H.device, dtype=H.dtype)
    C_vecs = torch.zeros(n_heads, d_state, device=H.device, dtype=H.dtype)

    for h in range(n_heads):
        # SVD of Hankel matrix
        try:
            U, S, Vh = torch.linalg.svd(H[h], full_matrices=False)
        except Exception:
            # Fallback: random init if SVD fails (degenerate matrix)
            print(f"  [ACSI] WARNING: SVD failed for head {h}, using random init")
            B_vecs[h] = torch.randn(d_state, device=H.device) * 0.02
            C_vecs[h] = torch.randn(d_state, device=H.device) * 0.02
            A_scalars[h] = -math.log(h + 1)  # Standard A_log init
            continue

        # Truncate to effective rank (ignore near-zero singular values)
        rank = min(d_state, (S > S[0] * 1e-6).sum().item())
        if rank < 1:
            rank = 1

        S_sqrt = S[:rank].sqrt()
        S_sqrt_inv = 1.0 / (S_sqrt + 1e-10)

        # Observability and controllability factors
        O = U[:, :rank] * S_sqrt[None, :]          # (d_state, rank)
        Ctrl = S_sqrt[:, None] * Vh[:rank, :]        # (rank, d_state)

        # System matrix A_hat = O^+ * H_shift * Ctrl^+
        # Using the balanced realization formula
        A_hat = torch.diag(S_sqrt_inv) @ U[:, :rank].T @ H_shift[h] @ Vh[:rank, :].T @ torch.diag(S_sqrt_inv)

        # Extract dominant eigenvalue for scalar A
        try:
            eigenvalues = torch.linalg.eigvals(A_hat[:rank, :rank])
            # Dominant eigenvalue = largest absolute value
            eig_abs = eigenvalues.abs()
            dominant_idx = eig_abs.argmax()
            dominant_eig = eigenvalues[dominant_idx].real
            # Clamp to stable range: A should be negative (decaying) for SSM stability
            # Our parameterization is A = -exp(A_log), so A is always negative
            # The identified A_scalar should capture the decay rate
            A_scalar = dominant_eig.clamp(-0.999, -0.001)  # Stable range
            A_scalars[h] = A_scalar
        except Exception:
            A_scalars[h] = -math.exp(math.log(h + 1))  # Fallback

        # B = first column of controllability matrix, normalized
        B_vecs[h, :rank] = Ctrl[:, 0]
        if rank < d_state:
            B_vecs[h, rank:] = torch.randn(d_state - rank, device=H.device) * 0.01

        # C = first row of observability matrix, normalized
        C_vecs[h, :rank] = O[0, :]
        if rank < d_state:
            C_vecs[h, rank:] = torch.randn(d_state - rank, device=H.device) * 0.01

    return A_scalars, B_vecs, C_vecs


@torch.no_grad()
def acsi_init_model(model, dataloader, device: str = 'cuda', max_batches: int = 4):
    """Apply ACSI initialization to all SSM layers in the model.

    Runs a few batches of data through the model, collects SSM input statistics
    at each layer, and applies Ho-Kalman system identification to initialize
    SSM parameters.

    Args:
        model: ModelV2 instance
        dataloader: DataLoader providing input_ids batches
        device: Device for computation
        max_batches: Number of batches to use for statistics (more = better estimates)
    """
    model.eval()
    print(f"[ACSI] Starting system identification initialization...")

    # Collect a few batches of input data
    input_batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        elif isinstance(batch, dict):
            input_ids = batch['input_ids']
        else:
            input_ids = batch
        input_batches.append(input_ids[:, :model.config.seq_len].to(device))

    if not input_batches:
        print("[ACSI] WARNING: No data batches available, skipping ACSI init")
        return

    all_inputs = torch.cat(input_batches, dim=0)  # (total_batch, seq_len)
    print(f"[ACSI] Collected {all_inputs.shape[0]} samples, seq_len={all_inputs.shape[1]}")

    # Get embeddings
    x = model.embedding(all_inputs)
    if hasattr(model, 'embed_dropout'):
        x = model.embed_dropout(x)

    n_initialized = 0

    # Process each layer
    for layer_idx, layer in enumerate(model.layers):
        block = layer.block if hasattr(layer, 'block') else None

        # Skip non-SSM layers (SWA layers, RWKV, etc.)
        if block is None:
            continue
        if not hasattr(block, 'A_log') and not hasattr(block, 'A_shadow'):
            continue

        is_lti = getattr(block, 'lti_mode', False)
        n_heads = block.n_heads
        d_state = block.d_state
        d_inner = block.d_inner
        head_dim = d_inner // n_heads

        # Forward through norm
        norm = layer.norm if hasattr(layer, 'norm') else None
        x_normed = norm(x) if norm is not None else x

        # Forward through in_proj and conv1d to get SSM input
        xz = block.in_proj(x_normed)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = x_inner.transpose(1, 2)
        seq_len = x_inner.shape[1]
        x_conv = block.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = block.act_fn(x_conv)

        # Reshape to (batch, seq_len, n_heads, head_dim)
        batch_size = x_conv.shape[0]
        x_heads = x_conv.view(batch_size, seq_len, n_heads, head_dim).float()

        # Compute auto-correlation
        max_lag = 2 * d_state
        R = compute_autocorrelation(x_heads, max_lag)

        # CAUSAL-BACKDOOR-DECONFOUNDING (R19-C):
        # Residual connection adds x to block output → observed R(0) = R_true(0) + Var(x_normed).
        # In normalized coordinates Var ≈ 1/n_heads per head. Subtract to recover R_true.
        # Only R(k=0) is affected; R(k>0) is unbiased (residual is memoryless at lag>0).
        if layer_idx > 0:
            R[:, 0] = (R[:, 0] - 1.0 / n_heads).clamp(min=1e-6)
            print(f"  [ACSI-DECONF] Layer {layer_idx}: R(0) corrected by -1/{n_heads}")

        # Build Hankel matrices
        H, H_shift = build_hankel(R, d_state)

        # Ho-Kalman realization
        A_scalars, B_vecs, C_vecs = ho_kalman_realization(H, H_shift, d_state)

        # Apply to SSM parameters
        if is_lti and hasattr(block, 'B_fixed') and hasattr(block, 'C_fixed'):
            # LTI: full initialization (A, B, C)
            block.B_fixed.data.copy_(B_vecs)
            block.C_fixed.data.copy_(C_vecs)

            if hasattr(block, 'A_shadow') and block.use_tscs:
                # TSCS: A is ternary, map identified A to nearest {-1, 0, +1}
                A_ternary = torch.round(A_scalars).clamp(-1, 1)
                block.A_shadow.data.copy_(A_ternary)
                print(f"  [ACSI] Layer {layer_idx} (LTI+TSCS): A_ternary={A_ternary.tolist()}, "
                      f"B_norm={B_vecs.norm(dim=-1).tolist()}, C_norm={C_vecs.norm(dim=-1).tolist()}")
            else:
                # Standard A parameterization: A = -exp(A_log)
                # Identified A_scalars are negative. A_log = log(-A_scalar)
                A_log_vals = torch.log((-A_scalars).clamp(min=1e-6))
                block.A_log.data.copy_(A_log_vals)
                print(f"  [ACSI] Layer {layer_idx} (LTI): A_log={A_log_vals.tolist()}, "
                      f"B_norm={B_vecs.norm(dim=-1).mean():.4f}, C_norm={C_vecs.norm(dim=-1).mean():.4f}")
            n_initialized += 1

        elif hasattr(block, 'A_log'):
            # Selective: A_log initialization only
            A_log_vals = torch.log((-A_scalars).clamp(min=1e-6))
            block.A_log.data.copy_(A_log_vals)
            print(f"  [ACSI] Layer {layer_idx} (selective): A_log={A_log_vals.tolist()}")
            n_initialized += 1

        elif hasattr(block, 'A_shadow') and block.use_tscs:
            # TSCS selective: map to ternary
            A_ternary = torch.round(A_scalars).clamp(-1, 1)
            block.A_shadow.data.copy_(A_ternary)
            print(f"  [ACSI] Layer {layer_idx} (selective+TSCS): A_ternary={A_ternary.tolist()}")
            n_initialized += 1

        # Forward through the full layer to update x for next layer
        # (needed so next layer's ACSI uses correct input statistics)
        with torch.no_grad():
            layer_out = layer(x)
            if isinstance(layer_out, tuple):
                x = layer_out[0]
            else:
                x = layer_out

    model.train()
    print(f"[ACSI] Initialized {n_initialized} SSM layers via system identification")
