"""OLS recalibration of out_proj weights after FP stage.

Usage (inside FP loop, after training layer k):
    from src.model_v11.training.ols import ols_recalibrate_out_proj

    improvement = ols_recalibrate_out_proj(
        model, layer_idx=k, dataloader=dataloader,
        device=device, n_batches=5
    )
    print(f"[OLS] layer {k}: improvement={improvement:+.4f} nats")
"""

import torch
import torch.nn.functional as F


def ols_recalibrate_out_proj(
    model,
    layer_idx: int,
    dataloader,
    device: str,
    n_batches: int = 5,
    ridge: float = 1e-4,
) -> float:
    """Recalibrate out_proj for layer layer_idx using OLS.

    Collects (H_in_to_out_proj, Y_target) pairs across n_batches,
    solves W_new = lstsq(H, Y)^T, overwrites out_proj.weight in-place.

    Assumes:
    - Layers 0..layer_idx are on device (or will be moved here temporarily).
    - Layers layer_idx+1..n-1 are frozen (FP convention).
    - lm_head, final_norm, embedding are on device.

    Args:
        model: ModelV2 instance.
        layer_idx: Layer index (0-indexed) to recalibrate.
        dataloader: Iterable yielding (x, targets) tuples (same interface as ByteDataLoader).
        device: Device string ("cuda" or "cpu").
        n_batches: Number of batches to collect for OLS solve.
        ridge: Ridge regularization parameter for fallback normal equations.

    Returns:
        Improvement signal (positive = better). 0.0 if out_proj not found on this layer type.
    """

    layer = model.layers[layer_idx]

    if layer.is_rwkv7:
        return 0.0

    if not hasattr(layer.block, 'out_proj'):
        return 0.0

    n_layers = len(model.layers)

    model.to(device)

    for l in model.layers:
        l.to(device)

    H_list, Y_list = [], []

    model.eval()
    for step, (x, targets) in enumerate(dataloader):
        if step >= n_batches:
            break

        x, targets = x.to(device), targets.to(device)

        with torch.no_grad():
            saved_h = [model.embedding(x)]
            for i in range(n_layers):
                saved_h.append(model.layers[i](saved_h[-1]))

        h_final = saved_h[-1].detach().clone().float().requires_grad_(True)
        h_norm = model.final_norm(h_final)
        if model.lm_head is not None:
            logits = model.lm_head(h_norm).float()
        else:
            logits = F.linear(h_norm, model.embedding.weight).float()
        logits = logits[0] if isinstance(logits, tuple) else logits
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss.backward()
        grad_h = h_final.grad.clone()

        for i in range(n_layers - 1, layer_idx, -1):
            h_in = saved_h[i].detach().clone().float().requires_grad_(True)
            h_out = model.layers[i](h_in)
            h_out.backward(grad_h)
            grad_h = h_in.grad.clone()

        h_buf = {}

        def hook(m, inp, out):
            h_buf["h_in"] = inp[0].detach().float().clone()

        handle = model.layers[layer_idx].block.out_proj.register_forward_hook(hook)
        with torch.no_grad():
            _ = model.layers[layer_idx](saved_h[layer_idx])
        handle.remove()

        H_list.append(h_buf["h_in"].reshape(-1, h_buf["h_in"].shape[-1]))
        Y_list.append(grad_h.reshape(-1, grad_h.shape[-1]))

        torch.cuda.empty_cache()

    H = torch.cat(H_list, dim=0)
    Y = torch.cat(Y_list, dim=0)

    try:
        result = torch.linalg.lstsq(H.cpu(), Y.cpu(), driver="gelsd")
        W_new = result.solution.T
    except Exception:
        A = H.T @ H + ridge * torch.eye(H.shape[1], dtype=H.dtype)
        B = H.T @ Y
        W_new = torch.linalg.solve(A, B).T

    if not torch.isfinite(W_new).all():
        print(f"[OLS] layer {layer_idx}: non-finite solution — skipping")
        for l in model.layers:
            l.cpu()
        torch.cuda.empty_cache()
        return 0.0

    norm_before = layer.block.out_proj.weight.data.norm().item()
    target_dtype = layer.block.out_proj.weight.dtype
    layer.block.out_proj.weight.data.copy_(W_new.to(target_dtype))
    norm_after = W_new.norm().item()

    print(
        f"[OLS] layer {layer_idx} recalibrate — "
        f"H={H.shape} Y={Y.shape} "
        f"norm_before={norm_before:.4f} norm_after={norm_after:.4f}"
    )

    for l in model.layers:
        l.cpu()

    torch.cuda.empty_cache()

    improvement = norm_before - norm_after
    return improvement
