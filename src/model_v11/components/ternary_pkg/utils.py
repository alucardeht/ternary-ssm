import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_antiwindup(model, k_aw=0.05):
    """Anti-windup for shadow weights (SQ-03 from research Round 3).

    Pulls continuous shadow weights toward their quantized ternary values,
    preventing drift into regions where gradients produce no ternary change.
    Analogous to integrator anti-windup in PID control theory.

    Args:
        model: The model containing TernaryLinear layers
        k_aw: Anti-windup gain (0.01-0.1 typical). Higher = stronger pull.
    """
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "group_size") and hasattr(module, "weight"):
                w = module.weight.data
                gs = module.group_size
                original_shape = w.shape

                flat = w.reshape(-1)
                pad_len = (gs - flat.numel() % gs) % gs
                if pad_len > 0:
                    flat = F.pad(flat, (0, pad_len))
                grouped = flat.reshape(-1, gs)

                scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                normalized = grouped / scales
                abs_norm = normalized.abs()
                threshold = getattr(module, "threshold_coeff", 0.7) * abs_norm.mean(
                    dim=-1, keepdim=True
                )
                quantized = normalized.sign() * (abs_norm > threshold).float()

                target = quantized * scales

                w_grouped = grouped.clone()
                drift = w_grouped - target
                w_grouped.sub_(k_aw * drift)

                w_flat = w_grouped.reshape(-1)
                if pad_len > 0:
                    w_flat = w_flat[: w.numel()]
                w.copy_(w_flat.reshape(original_shape))


def compute_angular_boundary_reg(
    model, angular_lambda=0.1, boundary_lambda=0.01, eps=1e-8
):
    """Compute angular and boundary-repulsive regularization for ternary layers.

    Angular: L = angular_lambda * mean(1 - cos_sim(Q(W).detach(), W)) per row, per layer.
    Boundary: L = -boundary_lambda * mean(log(d_boundary + eps)) where d is distance to
              nearest quantization threshold in raw weight space.

    Returns (angular_loss, boundary_loss, n_layers). Losses include lambda scaling.
    """
    angular_losses = []
    boundary_losses = []

    for module in model.modules():
        if not (
            hasattr(module, "quantize_enabled")
            and module.quantize_enabled
            and hasattr(module, "weight")
        ):
            continue

        W = module.weight
        gs = getattr(module, "group_size", 128)
        t_coeff = getattr(module, "threshold_coeff", 0.7)

        if angular_lambda > 0:
            with torch.no_grad():
                orig_shape = W.shape
                flat = W.detach().reshape(-1)
                pad_len = (gs - flat.numel() % gs) % gs
                if pad_len > 0:
                    flat = F.pad(flat, (0, pad_len))
                grouped = flat.reshape(-1, gs)
                scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                normalized = grouped / scales
                abs_norm = normalized.abs()
                threshold = t_coeff * abs_norm.mean(dim=-1, keepdim=True)
                quantized = normalized.sign() * (abs_norm > threshold).float()
                q_flat = (quantized * scales).reshape(-1)[: W.numel()]
                q_w = q_flat.reshape(orig_shape)

            cos = F.cosine_similarity(q_w, W, dim=1)
            angular_losses.append((1.0 - cos).mean())

        if boundary_lambda > 0:
            with torch.no_grad():
                flat_d = W.detach().reshape(-1)
                pad_len_d = (gs - flat_d.numel() % gs) % gs
                if pad_len_d > 0:
                    flat_d = F.pad(flat_d, (0, pad_len_d))
                grouped_d = flat_d.reshape(-1, gs)
                scales_d = (
                    grouped_d.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                )
                norm_d = grouped_d / scales_d
                threshold_d = t_coeff * norm_d.abs().mean(dim=-1, keepdim=True)
                boundary_raw = (
                    (threshold_d * scales_d)
                    .expand(-1, gs)
                    .reshape(-1)[: W.numel()]
                    .reshape(W.shape)
                )

            d_pos = (W - boundary_raw).abs()
            d_neg = (W + boundary_raw).abs()
            d_boundary = torch.minimum(d_pos, d_neg)
            boundary_losses.append(-torch.log(d_boundary + eps).mean())

    n_layers = max(len(angular_losses), len(boundary_losses), 0)

    if angular_losses:
        total_angular = angular_lambda * torch.stack(angular_losses).mean()
    else:
        device = next(model.parameters()).device
        total_angular = torch.tensor(0.0, device=device)

    if boundary_losses:
        total_boundary = boundary_lambda * torch.stack(boundary_losses).mean()
    else:
        device = next(model.parameters()).device
        total_boundary = torch.tensor(0.0, device=device)

    return total_angular, total_boundary, n_layers


def compute_tdo_loss(
    model, tdo_lambda=0.01, tdo_target=(1 / 3, 1 / 3, 1 / 3), temperature=10.0
):
    """Ternary Distribution Objective - regularizer pushing shadow weights toward target distribution.

    Computes soft (differentiable) approximation of ternary weight distribution using sigmoid,
    then penalizes deviation from target via KL divergence. Gradients flow through sigmoid
    back into shadow weights.
    """
    kl_losses = []
    dist_neg_sum = 0.0
    dist_zero_sum = 0.0
    dist_pos_sum = 0.0
    n_layers = 0

    for module in model.modules():
        if not (
            hasattr(module, "quantize_enabled")
            and module.quantize_enabled
            and hasattr(module, "weight")
        ):
            continue

        W = module.weight
        gs = getattr(module, "group_size", 64)
        t_coeff = getattr(module, "threshold_coeff", 0.7)

        # Group-wise normalize (same math as TernaryQuantize.forward)
        flat = W.reshape(-1)
        pad_len = (gs - flat.numel() % gs) % gs
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, gs)
        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        # Threshold
        abs_norm = normalized.abs()
        threshold = t_coeff * abs_norm.mean(dim=-1, keepdim=True)

        # Differentiable soft distribution via sigmoid
        soft_above = torch.sigmoid(temperature * (abs_norm - threshold))
        soft_below = 1.0 - soft_above
        soft_sign_pos = torch.sigmoid(temperature * normalized)

        # Per-weight soft probabilities
        p_pos_w = soft_above * soft_sign_pos
        p_neg_w = soft_above * (1.0 - soft_sign_pos)
        p_zero_w = soft_below

        # Layer-wide distribution
        p_neg = p_neg_w.mean()
        p_zero = p_zero_w.mean()
        p_pos = p_pos_w.mean()

        # KL divergence to target
        actual = torch.stack([p_neg, p_zero, p_pos]).clamp(min=1e-8)
        target_t = torch.tensor(tdo_target, device=W.device, dtype=W.dtype)
        kl = (actual * (actual.log() - target_t.log())).sum()

        kl_losses.append(kl)
        dist_neg_sum += p_neg.item()
        dist_zero_sum += p_zero.item()
        dist_pos_sum += p_pos.item()
        n_layers += 1

    if n_layers == 0:
        dummy = torch.tensor(
            0.0, device=next(model.parameters()).device, requires_grad=True
        )
        return dummy, 0, {"neg": 0, "zero": 0, "pos": 0}

    tdo_loss = tdo_lambda * torch.stack(kl_losses).mean()
    stats = {
        "neg": dist_neg_sum / n_layers,
        "zero": dist_zero_sum / n_layers,
        "pos": dist_pos_sum / n_layers,
    }

    return tdo_loss, n_layers, stats


def compute_weight_entropy(model) -> list:
    """Compute Shannon entropy of ternary weight distribution per layer.

    Returns list of dicts with entropy metrics per TernaryLinear layer.
    H_max = log2(3) = 1.585 bits. Utilization < 0.8 indicates capacity loss.
    """
    results = []
    for name, module in model.named_modules():
        if not hasattr(module, "get_ternary_stats"):
            continue
        stats = module.get_ternary_stats()
        probs = [stats["negative"], stats["zero"], stats["positive"]]
        H = 0.0
        for p in probs:
            if p > 1e-10:
                H -= p * math.log2(p)
        H_max = math.log2(3)
        results.append(
            {
                "name": name,
                "entropy": H,
                "max_entropy": H_max,
                "utilization": H / H_max,
                "dist": (stats["negative"], stats["zero"], stats["positive"]),
            }
        )
    return results


def compute_commitment_rate(model, threshold_margin=0.1) -> list:
    """Compute fraction of weights committed (far from threshold) per layer.

    'Uncommitted' = shadow weight's |w_normalized| is within threshold_margin
    of the quantization threshold. These weights may flip with small gradient changes.
    """
    results = []
    for name, module in model.named_modules():
        if not hasattr(module, "threshold_coeff") or not hasattr(module, "weight"):
            continue
        if not hasattr(module, "group_size"):
            continue
        with torch.no_grad():
            w = module.weight
            group_size = module.group_size
            t_coeff = module.threshold_coeff

            flat = w.reshape(-1)
            pad_len = (group_size - flat.numel() % group_size) % group_size
            if pad_len > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_len))
            grouped = flat.reshape(-1, group_size)

            scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            normalized = grouped / scales
            abs_normalized = normalized.abs()
            threshold = t_coeff * abs_normalized.mean(dim=-1, keepdim=True)

            dist_to_threshold = (abs_normalized - threshold).abs()
            margin = threshold_margin * threshold
            uncommitted = (dist_to_threshold < margin).float()

            total = w.numel()
            n_uncommitted = uncommitted.sum().item()
            if pad_len > 0:
                n_uncommitted = uncommitted.reshape(-1)[: w.numel()].sum().item()

            results.append(
                {
                    "name": name,
                    "committed_frac": 1.0 - n_uncommitted / total,
                    "uncommitted_frac": n_uncommitted / total,
                    "mean_dist_to_threshold": dist_to_threshold.reshape(-1)[: w.numel()]
                    .mean()
                    .item()
                    if pad_len > 0
                    else dist_to_threshold.mean().item(),
                    "total_weights": total,
                }
            )
    return results


def compute_margin_mask(
    weight: torch.Tensor, group_size: int, threshold_coeff: float, delta: float
) -> torch.Tensor:
    """Compute binary mask of weights near quantization boundary.

    Returns: bool tensor same shape as weight. True = critical (near boundary).
    """
    with torch.no_grad():
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        pad_len = (group_size - flat.numel() % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, group_size)

        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales
        abs_normalized = normalized.abs()
        threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)

        dist_to_threshold = (abs_normalized - threshold).abs()
        margin_threshold = delta * threshold
        critical = dist_to_threshold < margin_threshold

        critical_flat = critical.reshape(-1)
        if pad_len > 0:
            critical_flat = critical_flat[: weight.numel()]
        return critical_flat.reshape(orig_shape)


def init_gradient_recyclers(model, beta: float = 0.99) -> dict:
    """Initialize EMA gradient storage for all TernaryLinear weight parameters.

    Returns dict mapping param name -> {'ema': Tensor, 'beta': float, 'initialized': bool}
    """
    recyclers = {}
    for name, module in model.named_modules():
        if not hasattr(module, "threshold_coeff") or not hasattr(module, "weight"):
            continue
        if not hasattr(module, "group_size"):
            continue
        param_name = name + ".weight"
        recyclers[param_name] = {
            "ema": torch.zeros_like(module.weight.data),
            "beta": beta,
            "initialized": False,
            "group_size": module.group_size,
            "threshold_coeff": module.threshold_coeff,
        }
    print(
        f"[MARGIN-SPARSE] Initialized gradient recyclers for {len(recyclers)} ternary layers "
        f"({sum(r['ema'].numel() for r in recyclers.values())} params tracked)"
    )
    return recyclers


def apply_gradient_recycling(
    model,
    recyclers: dict,
    global_step: int,
    refresh_interval: int,
    delta: float,
    log_interval: int = 100,
) -> None:
    """Replace far-from-boundary gradients with EMA approximation.

    Call AFTER backward() and unscale, BEFORE grad clipping and optimizer.step().
    """
    is_refresh = global_step % refresh_interval == 0
    total_critical = 0
    total_params = 0

    for name, param in model.named_parameters():
        if name not in recyclers or param.grad is None:
            continue

        rec = recyclers[name]
        grad = param.grad.data

        if is_refresh or not rec["initialized"]:
            rec["ema"].mul_(rec["beta"]).add_(grad, alpha=1.0 - rec["beta"])
            rec["initialized"] = True
        else:
            critical_mask = compute_margin_mask(
                param.data, rec["group_size"], rec["threshold_coeff"], delta
            )
            critical_float = critical_mask.float()

            param.grad.data = (
                critical_float * grad + (1.0 - critical_float) * rec["ema"]
            )

            total_critical += critical_mask.sum().item()
            total_params += param.numel()

    if not torch.compiler.is_compiling() and global_step % log_interval == 0:
        if is_refresh:
            print(
                f"  [MARGIN-SPARSE] step={global_step} REFRESH (full gradient, EMA updated)"
            )
        elif total_params > 0:
            crit_pct = 100.0 * total_critical / total_params
            print(
                f"  [MARGIN-SPARSE] step={global_step} critical={crit_pct:.1f}% "
                f"({int(total_critical)}/{total_params} weights near boundary)"
            )
