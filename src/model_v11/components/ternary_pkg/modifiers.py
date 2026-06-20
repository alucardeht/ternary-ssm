import math
import torch
import torch.nn as nn


def enable_dendritic_gating(model, mode: str):
    """Enable dendritic gating on all TernaryLinear layers post-construction."""
    assert mode in ("pre", "post"), f"Invalid dendritic mode: '{mode}'"
    added = 0
    total_gate_params = 0
    for module in model.modules():
        if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
            out_features = module.weight.shape[0]
            module.dendritic_mode = mode
            if mode == "pre":
                module.gate_pre = nn.Parameter(
                    torch.ones(
                        out_features,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                )
                total_gate_params += out_features
            elif mode == "post":
                module.gate_post = nn.Parameter(
                    torch.ones(
                        out_features,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                )
                total_gate_params += out_features
            added += 1
    print(
        f"[Dendritic] Enabled mode='{mode}' on {added} layers, "
        f"{total_gate_params:,} gate params ({total_gate_params * 4 / 1024:.1f} KB FP32)"
    )
    return added, total_gate_params


def enable_sigma_delta(model, error_clamp: float = 2.0):
    """Enable sigma-delta noise shaping on all TernaryLinear layers."""
    enabled = 0
    for module in model.modules():
        if hasattr(module, "ste_type") and hasattr(module, "quantize_enabled"):
            module.ste_type = "sigma_delta"
            module.sd_error_clamp = error_clamp
            enabled += 1
    print(f"[SigmaDelta] Enabled on {enabled} layers, error_clamp={error_clamp}")
    return enabled


def enable_bias_correction(model, momentum: float = 0.99, variant: str = "naive"):
    """Enable bias-corrected STE (BSTE) on all TernaryLinear layers (Phase 21.D, TQ-02).

    STE has a systematic bias: gradients passed to weights deep in quantization bins
    are pure noise that cannot cause ternary value changes. BSTE maintains a running
    EMA of this bias and subtracts it before the optimizer step.

    Variants:
        'naive': EMA of raw gradient. High-pass temporal filter on gradient signal.
        'error_weighted': EMA weighted by quantization depth. Only corrects weights
                         deep in bins (far from threshold), leaves boundary weights alone.

    Parameter-free (zero extra trainable parameters). Only adds a buffer for EMA state.
    Must call apply_bias_correction(model) between loss.backward() and optimizer.step().

    Args:
        model: The model containing TernaryLinear layers
        momentum: EMA momentum (0.99 = ~100-step window)
        variant: 'naive' or 'error_weighted'
    Returns:
        Number of layers enabled
    """
    assert variant in ("naive", "error_weighted"), f"Invalid BSTE variant: '{variant}'"
    enabled = 0
    for module in model.modules():
        if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
            module.register_buffer("gradient_bias_ema", torch.zeros_like(module.weight))
            module.bste_momentum = momentum
            module.bste_variant = variant
            module.bste_enabled = True
            enabled += 1
    print(
        f"[BSTE] Enabled variant='{variant}' momentum={momentum} on {enabled} layers "
        f"(buffer size: {sum(m.gradient_bias_ema.numel() for m in model.modules() if hasattr(m, 'gradient_bias_ema')) * 4 / 1024:.1f} KB)"
    )
    return enabled


def apply_bias_correction(model):
    """Apply bias-corrected STE gradient correction (Phase 21.D).

    Call AFTER loss.backward() and BEFORE optimizer.step() in the training loop.

    For 'naive' variant: subtracts EMA of raw gradient (temporal high-pass filter).
    For 'error_weighted' variant: subtracts EMA of gradient weighted by quantization
    depth (only corrects weights deep in bins, preserves boundary gradients).

    Returns:
        Tuple of (correction_applied: bool, mean_correction_ratio: float)
    """
    correction_ratios = []
    for module in model.modules():
        if not (hasattr(module, "bste_enabled") and module.bste_enabled):
            continue
        if module.weight.grad is None:
            continue

        grad = module.weight.grad
        ema = module.gradient_bias_ema
        momentum = module.bste_momentum

        if module.bste_variant == "error_weighted":
            with torch.no_grad():
                w = module.weight.data
                gs = getattr(module, "group_size", 128)
                t_coeff = getattr(module, "threshold_coeff", 0.7)
                orig_shape = w.shape
                flat = w.reshape(-1)
                pad_len = (gs - flat.numel() % gs) % gs
                if pad_len > 0:
                    flat = torch.nn.functional.pad(flat, (0, pad_len))
                grouped = flat.reshape(-1, gs)
                scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                normalized = grouped / scales
                abs_norm = normalized.abs()
                threshold = t_coeff * abs_norm.mean(dim=-1, keepdim=True)
                quantized_scaled = (
                    normalized.sign() * (abs_norm > threshold).float() * scales
                )
                q_flat = quantized_scaled.reshape(-1)[: w.numel()]
                q_w = q_flat.reshape(orig_shape)

                quant_error = (w - q_w).abs()
                max_err = quant_error.max().clamp(min=1e-8)
                depth_mask = (quant_error / max_err).clamp(0, 1)

            ema.mul_(momentum).add_(grad * depth_mask, alpha=1 - momentum)
        else:
            ema.mul_(momentum).add_(grad, alpha=1 - momentum)

        with torch.no_grad():
            grad_norm = grad.abs().mean().clamp(min=1e-10)
            correction_ratio = ema.abs().mean() / grad_norm
            correction_ratios.append(correction_ratio.item())

        module.weight.grad = grad - ema

    if correction_ratios:
        return True, sum(correction_ratios) / len(correction_ratios)
    return False, 0.0


def enable_block_diagonal(model, block_size: int = 4):
    """Enable block-diagonal FP16 mixing on all TernaryLinear layers (Phase 22).

    After the ternary matmul, groups every block_size consecutive output neurons
    and mixes them with a learnable FP16 matrix. Restores singular value
    magnitude variation lost by ternary quantization.

    Storage overhead: block_size extra FP16 params per output neuron.
    For block_size=4: ~3.1% overhead at typical dimensions.

    Args:
        model: The model containing TernaryLinear layers
        block_size: Number of output neurons per mixing block (default 4)
    Returns:
        Tuple of (num_layers_enabled, total_mixing_params)
    """
    enabled = 0
    total_params = 0
    skipped = 0
    for module in model.modules():
        if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
            out_features = module.weight.shape[0]
            if out_features % block_size != 0:
                skipped += 1
                continue
            num_blocks = out_features // block_size
            module.block_size = block_size
            module.block_mixing = nn.Parameter(
                torch.eye(
                    block_size, device=module.weight.device, dtype=module.weight.dtype
                )
                .unsqueeze(0)
                .expand(num_blocks, -1, -1)
                .clone()
            )
            n_params = num_blocks * block_size * block_size
            total_params += n_params
            enabled += 1
    print(
        f"[BlockDiag] Enabled block_size={block_size} on {enabled} layers "
        f"({total_params:,} mixing params, {total_params * 4 / 1024:.1f} KB FP32), "
        f"skipped {skipped} layers (dim not divisible by {block_size})"
    )
    return enabled, total_params


def get_block_diagonal_stats(model):
    """Get block mixing deviation statistics for debug logging."""
    deviations = []
    for name, module in model.named_modules():
        if hasattr(module, "block_mixing") and module.block_mixing is not None:
            with torch.no_grad():
                C = module.block_mixing
                bs = C.shape[-1]
                identity = torch.eye(bs, device=C.device, dtype=C.dtype).unsqueeze(0)
                deviation = (C - identity).norm(dim=(-2, -1)).mean().item()
                deviations.append(deviation)
    if not deviations:
        return {"enabled": False}
    return {
        "enabled": True,
        "num_layers": len(deviations),
        "avg_deviation_from_identity": sum(deviations) / len(deviations),
        "max_deviation": max(deviations),
        "min_deviation": min(deviations),
    }


def enable_tpsq(model, beta: float = 1.0, momentum: float = 0.99):
    """Enable TPSQ (Phase-Space Quantization) on all TernaryLinear layers.

    Maintains running statistics of input variance and uses it to adaptively
    adjust quantization thresholds.

    Args:
        model: The model containing TernaryLinear layers
        beta: TPSQ sensitivity to input variance (default: 1.0)
        momentum: EMA momentum for running variance (default: 0.99)
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_tpsq"):
            module.use_tpsq = True
            module.tpsq_beta = beta
            module.tpsq_momentum = momentum
            # Register buffer only if not already registered (handles multiple enable calls)
            if not hasattr(module, "_tpsq_sigma_sq"):
                device = module.weight.device
                dtype = module.weight.dtype
                group_size = module.group_size
                n_groups = (module.weight.numel() + group_size - 1) // group_size
                module.register_buffer(
                    "_tpsq_sigma_sq", torch.ones(n_groups, device=device, dtype=dtype)
                )
            enabled += 1
    print(f"[TPSQ] Enabled on {enabled} layers (beta={beta}, momentum={momentum})")
    return enabled


def enable_tcd(model, beta_init: float = None):
    """Enable TCD (Ternary Critical Dynamics) on all TernaryLinear layers.

    Sets ste_type='tcd' and initializes beta_c = sqrt(d_model/3) for critical damping.

    Args:
        model: The model containing TernaryLinear layers
        beta_init: Initial beta value. If None, uses sqrt(d_model/3)
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if hasattr(module, "tcd_beta") and hasattr(module, "ste_type"):
            if beta_init is None:
                d_model = getattr(module, "out_features", 256)
                beta = math.sqrt(d_model / 3.0)
            else:
                beta = beta_init
            module.tcd_beta = max(0.1, min(beta, 100.0))
            module.ste_type = "tcd"
            enabled += 1
    print(f"[TCD] Enabled on {enabled} layers (ste_type='tcd')")
    return enabled


def enable_mdts(model, temperature: float = 1.0, anneal: bool = True):
    """Enable MDTS (Mirror Descent Ternary Solver) on all TernaryLinear layers.

    Initializes logits from current shadow weights, confident toward existing ternary.
    Temperature anneals from 1.0 to 0.1 over training if anneal=True.

    Args:
        model: The model containing TernaryLinear layers
        temperature: Initial temperature (default: 1.0)
        anneal: If True, anneal temperature 1.0 -> 0.1 over training
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if not (hasattr(module, "use_mdts") and hasattr(module, "weight")):
            continue

        W = module.weight.data
        gs = module.group_size
        device = W.device
        dtype = W.dtype

        orig_shape = W.shape
        flat = W.reshape(-1)
        pad_len = (gs - flat.numel() % gs) % gs
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len))

        grouped = flat.reshape(-1, gs)
        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales

        abs_norm = normalized.abs()
        logits_init = torch.zeros(grouped.shape[0], 3, device=device, dtype=dtype)

        pos_mask = (normalized > 0.5).float()
        neg_mask = (normalized < -0.5).float()
        zero_mask = 1.0 - pos_mask - neg_mask

        logits_init[:, 0] = (
            neg_mask.sum(dim=-1) / (neg_mask.sum(dim=-1) + 1e-8)
        ).clamp(min=1e-4)
        logits_init[:, 1] = (
            zero_mask.sum(dim=-1) / (zero_mask.sum(dim=-1) + 1e-8)
        ).clamp(min=1e-4)
        logits_init[:, 2] = (
            pos_mask.sum(dim=-1) / (pos_mask.sum(dim=-1) + 1e-4)
        ).clamp(min=1e-4)

        logits_init = torch.log(logits_init + 1e-10)

        module._mdts_logits = nn.Parameter(logits_init)
        if hasattr(module, "_mdts_scales"):
            delattr(module, "_mdts_scales")
        module.register_buffer("_mdts_scales", scales.squeeze(-1))

        module.use_mdts = True
        module.mdts_temperature = temperature
        module.mdts_anneal = anneal
        module._mdts_call_count = 0

        enabled += 1

    anneal_str = "with annealing" if anneal else "constant"
    print(
        f"[MDTS] Enabled on {enabled} layers (temperature={temperature} {anneal_str})"
    )
    return enabled


def enable_gumbel_noise(model, temperature: float = 0.1, anneal: bool = False):
    """Enable Gumbel+STE noise injection on all TernaryLinear layers.

    Adds Gumbel noise during training to improve exploration of weight space
    and gradient flow around quantization boundaries.

    Args:
        model: The model containing TernaryLinear layers
        temperature: Gumbel noise scale (default: 0.1)
        anneal: If True, decay temperature over training (default: False)
    Returns:
        Number of layers enabled
    """
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_gumbel_noise"):
            module.use_gumbel_noise = True
            module.gumbel_temperature = temperature
            module.gumbel_anneal = anneal
            enabled += 1
    anneal_str = "with annealing" if anneal else "constant"
    print(
        f"[GumbelNoise] Enabled on {enabled} layers (temperature={temperature} {anneal_str})"
    )
    return enabled


def enable_stp(model, sigma_ratio: float = 0.1):
    """Enable STP (Stochastic Threshold Perturbation) on all TernaryLinear layers."""
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_stp"):
            module.use_stp = True
            module.stp_sigma_ratio = sigma_ratio
            enabled += 1
    print(f"[STP] Enabled on {enabled} layers (sigma_ratio={sigma_ratio})")
    return enabled


def enable_ctce(model, beta: float = 1.5, delta_ratio: float = 0.5):
    """Enable CTCE on all TernaryLinear layers. Amplifies gradients near threshold (factor >= 1.0)."""
    enabled = 0
    for module in model.modules():
        if hasattr(module, "use_ctce"):
            module.use_ctce = True
            module.ctce_beta = beta
            module.ctce_delta_ratio = delta_ratio
            enabled += 1
    print(
        f"[CTCE] Enabled on {enabled} layers (beta={beta}, delta_ratio={delta_ratio})"
    )
    return enabled


def enable_dual_rate(
    model, peak: float = 2.5, floor: float = 0.1, delta_ratio: float = 0.5
):
    """Enable Dual-Rate STE on all TernaryLinear layers.
    Combines CTCE amplification with far-from-boundary damping.
    Disables CTCE if active (Dual-Rate supersedes it).
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "use_stp"):
            module.use_dual_rate = True
            module.dual_rate_peak = peak
            module.dual_rate_floor = floor
            module.dual_rate_delta_ratio = delta_ratio
            module.use_ctce = False  # Dual-Rate supersedes CTCE
            count += 1
    print(
        f"[DualRate] Enabled on {count} layers (peak={peak}, floor={floor}, delta_ratio={delta_ratio})"
    )
    return count


def enable_gaste(model, sigma_g: float = 0.3):
    """Enable GASTE (Group-Aware STE) on all TernaryLinear layers."""
    count = 0
    for module in model.modules():
        if hasattr(module, "use_stp"):
            module.use_gaste = True
            module.gaste_sigma_g = sigma_g
            count += 1
    print(f"[GASTE] Enabled on {count} layers (sigma_g={sigma_g})")
    return count


def enable_rlec(model, rank: int = 4, min_size: int = 0):
    """Enable RLEC (Residual Low-Rank Error Compensation) on all TernaryLinear layers.

    RLEC adds FP16 low-rank correction: y = Q(W)*x + A*(B*x)
    A initialized to zeros (starts as pure ternary), B to scaled random.
    """
    import math

    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "use_stp"):
            out_f, in_f = module.weight.shape
            if min_size > 0 and min(out_f, in_f) < min_size:
                continue
            # RLEC initialization: A=zeros (start pure ternary), B=scaled random
            module.correction_A = torch.nn.Parameter(
                torch.zeros(
                    out_f, rank, device=module.weight.device, dtype=torch.float32
                )
            )
            module.correction_B = torch.nn.Parameter(
                torch.randn(
                    rank, in_f, device=module.weight.device, dtype=torch.float32
                )
                * (1.0 / math.sqrt(in_f))
            )
            module._rlec_call_count = 0
            count += 1
    total_params = sum(
        m.correction_A.numel() + m.correction_B.numel()
        for m in model.modules()
        if hasattr(m, "correction_A")
        and m.correction_A is not None
        and m.correction_A.shape[-1] == rank
    )
    print(
        f"[RLEC] Enabled on {count} layers, rank={rank}, added {total_params:,} FP16 params ({total_params * 2 / 1024:.1f} KB)"
    )
    return count
