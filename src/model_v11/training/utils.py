import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable
import os
import math
import gc


def compile_model(model, mode: str = "reduce-overhead", backend: str = "inductor"):

    try:
        compiled = torch.compile(model, mode=mode, backend=backend)
        print(
            f"[compile] Model compiled with mode={mode}, backend={backend}"
        )
        return compiled
    except Exception as e:
        print(f"[compile] Failed: {e}. Using uncompiled model.")
        return model


def get_optimal_dataloader_kwargs(
    num_cpu_cores: Optional[int] = None,
) -> Dict:

    if num_cpu_cores is None:
        num_cpu_cores = os.cpu_count() or 4

    return {
        "pin_memory": torch.cuda.is_available(),
        "num_workers": min(4, num_cpu_cores),
        "prefetch_factor": 2,
        "persistent_workers": True,
    }


class BatchSizeConfig:

    def __init__(
        self,
        target_batch: int = 128,
        micro_batch: int = 16,
        grad_accum_steps: Optional[int] = None,
    ):
        self.target_batch = target_batch
        self.micro_batch = micro_batch

        if grad_accum_steps is not None:
            self.grad_accum_steps = grad_accum_steps
        else:
            self.grad_accum_steps = max(1, target_batch // micro_batch)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum_steps

    def __repr__(self) -> str:
        return (
            f"BatchSizeConfig(micro={self.micro_batch}, "
            f"accum={self.grad_accum_steps}, "
            f"effective={self.effective_batch})"
        )


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def compute_optimal_batch_size(
    model: nn.Module, seq_len: int, vram_budget_mb: int, device: torch.device
) -> int:
    if device.type != "cuda":
        return 1

    torch.cuda.empty_cache()
    gc.collect()

    current_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    available_mb = vram_budget_mb - current_mb

    if available_mb <= 0:
        return 1

    param_count = sum(p.numel() for p in model.parameters())
    est_per_sample_mb = (param_count * 4 / 1024 / 1024) * 0.5

    if est_per_sample_mb <= 0:
        return 1

    optimal = max(1, int(available_mb / est_per_sample_mb))
    optimal = min(optimal, 64)

    return optimal


def _quantize_to_fp8(tensor: torch.Tensor) -> tuple:
    if tensor.numel() == 0:
        return tensor.to(torch.float8_e4m3fn), torch.tensor(1.0)
    amax = tensor.abs().max().clamp(min=1e-12)
    scale = amax / 448.0
    scaled = tensor / scale
    quantized = scaled.to(torch.float8_e4m3fn)
    return quantized, scale


def _dequantize_from_fp8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.float() * scale


def apply_topk_gradient_sparsification(
    model: nn.Module, sparsity: float = 0.9, error_buffers: dict = None,
):
    if error_buffers is None:
        return

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.data

        if name in error_buffers:
            grad = grad + error_buffers[name]

        k = max(1, int(grad.numel() * (1 - sparsity)))
        topk_vals, topk_idx = torch.topk(grad.abs().view(-1), k)

        mask = torch.zeros(grad.numel(), dtype=torch.bool, device=grad.device)
        mask[topk_idx] = True
        mask = mask.view_as(grad)

        sparse_grad = grad * mask.float()
        error_buffers[name] = (grad - sparse_grad).detach()

        param.grad.data.copy_(sparse_grad)


def compute_fisher_proxy(model) -> dict:
    fisher = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher[name] = param.grad.detach().square()
    return fisher


def apply_fisher_preconditioning(
    model, fisher_accum: dict, step: int, damping: float = 1e-4,
):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name not in fisher_accum:
            fisher_accum[name] = param.grad.detach().square()
        else:
            fisher_accum[name] = fisher_accum[name] + param.grad.detach().square()

        fisher_avg = fisher_accum[name] / max(step, 1)
        precond = 1.0 / (fisher_avg.sqrt() + damping)
        param.grad.mul_(precond)


def apply_sign_momentum(model, momentum_buffers: dict, beta: float = 0.9):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        if name not in momentum_buffers:
            momentum_buffers[name] = torch.zeros_like(param.data)

        buf = momentum_buffers[name]
        buf.mul_(beta).add_(param.grad, alpha=1 - beta)

        param.grad.copy_(buf.sign())


def apply_consensus_gradient_filter(model, consensus_alpha: float = 0.5):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad
        median = grad.median()
        mad = (grad - median).abs().median()

        if mad > 0:
            threshold = consensus_alpha * mad
            grad_centered = grad - median
            clipped = torch.clamp(grad_centered, -threshold, threshold) + median
            param.grad.copy_(clipped)


def compute_layer_sensitivity(model) -> dict:
    sensitivities = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher_diag = param.grad.detach().square().mean().item()
            sensitivities[name] = fisher_diag
    return sensitivities


def apply_adaptive_quantization_levels(model, sensitivity_threshold: float = 0.1):
    sensitivities = compute_layer_sensitivity(model)
    if not sensitivities:
        return

    max_sens = max(sensitivities.values())
    if max_sens == 0:
        return

    for name, param in model.named_parameters():
        if name in sensitivities:
            norm_sens = sensitivities[name] / max_sens
            if norm_sens > sensitivity_threshold:
                param._quant_levels = 5
            else:
                param._quant_levels = 3


def log_layer_sensitivity(model, step: int) -> dict:
    if step % 50 != 0:
        return {}

    sensitivities = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().norm().item()
        weight_norm = param.data.detach().norm().item()
        s_l = grad_norm / max(weight_norm, 1e-8)
        sensitivities[name] = s_l

    if not sensitivities:
        return {}

    sorted_items = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_items[:8]
    bot_n = sorted_items[-4:]

    print(f"\n[SAP] step={step} top sensitivity layers:")
    print(f"  {'layer':<52} {'S_l':>10}")
    print(f"  {'-'*52} {'-'*10}")
    for layer_name, s_l in top_n:
        print(f"  {layer_name:<52} {s_l:>10.4f}")
    print(f"  ...")
    print(f"[SAP] step={step} bottom sensitivity layers:")
    for layer_name, s_l in bot_n:
        print(f"  {layer_name:<52} {s_l:>10.4f}")

    return sensitivities


def compute_checkpoint_density(num_layers: int) -> list:
    density = []
    for i in range(num_layers):
        frac = i / max(num_layers - 1, 1)
        if frac < 0.25:
            density.append(True)
        elif frac < 0.75:
            density.append(i % 2 == 0)
        else:
            density.append(False)
    return density


def compute_optimal_training_config(
    param_count: int,
    vram_budget_mb: int,
    use_ternary: bool = True,
    seq_len: int = 512,
) -> dict:
    if use_ternary:
        model_bytes = param_count * 2
        shadow_bytes = param_count * 4
        optimizer_bytes = param_count * 8
        grad_bytes = param_count * 4
        total_static_mb = (
            model_bytes + shadow_bytes + optimizer_bytes + grad_bytes
        ) / (1024 * 1024)
    else:
        model_bytes = param_count * 2
        optimizer_bytes = param_count * 12
        grad_bytes = param_count * 2
        total_static_mb = (model_bytes + optimizer_bytes + grad_bytes) / (1024 * 1024)

    available_mb = vram_budget_mb - total_static_mb
    available_mb = max(available_mb, 100)

    d_model = int((param_count / 12) ** 0.5)
    est_act_per_sample_mb = (seq_len * d_model * 4 * 2) / (1024 * 1024)

    if est_act_per_sample_mb <= 0:
        est_act_per_sample_mb = 100

    batch_size = max(1, int(available_mb / est_act_per_sample_mb))
    batch_size = min(batch_size, 64)

    target_effective_batch = 32
    grad_accum = max(1, target_effective_batch // batch_size)

    return {
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "total_static_mb": round(total_static_mb, 1),
        "est_activation_mb_per_sample": round(est_act_per_sample_mb, 1),
        "available_for_activations_mb": round(available_mb, 1),
    }


def get_ssm_checkpoint_indices(seq_len: int) -> list:
    interval = max(1, int(math.sqrt(seq_len)))
    return list(range(0, seq_len, interval))


def compute_weight_snr(
    model, grad_sq_accum: dict, grad_accum: dict, step: int, min_steps: int = 50
) -> dict:
    snr = {}
    if step < min_steps:
        return snr

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        g = param.grad.detach()

        if name not in grad_accum:
            grad_accum[name] = g.clone()
            grad_sq_accum[name] = g.square()
        else:
            grad_accum[name].add_(g)
            grad_sq_accum[name].add_(g.square())

        mean_g = grad_accum[name] / step
        mean_g_sq = grad_sq_accum[name] / step
        var_g = mean_g_sq - mean_g.square()
        var_g = var_g.clamp(min=1e-10)

        snr[name] = mean_g.square() / var_g

    return snr


def freeze_low_snr_weights(model, snr: dict, threshold: float = 0.1):
    for name, param in model.named_parameters():
        if name in snr and param.grad is not None:
            param_snr = snr[name].median().item()
            if param_snr < threshold:
                param.grad.zero_()


@torch.no_grad()
def _apply_gradient_noise(model: nn.Module, step: int, eta: float, gamma: float):
    noise_std = eta / ((1 + step) ** gamma)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad) * noise_std)
    return noise_std


def compute_adaptive_seq_len(
    model: nn.Module, min_seq: int = 64, max_seq: int = 512
) -> int:
    from ..spectral import compute_spectral_radius, compute_safe_sequence_length

    safe_lengths = []
    for name, param in model.named_parameters():
        if "A_log" in name or "a_log" in name:
            sr = compute_spectral_radius(param)
            sl = compute_safe_sequence_length(sr)
            safe_lengths.append(sl)

    if not safe_lengths:
        return max_seq

    safe_lengths.sort()
    median_safe = safe_lengths[len(safe_lengths) // 2]

    return max(min_seq, min(median_safe, max_seq))


def compute_batch_geometry(
    seq_len: int, base_batch: int = 16, base_seq: int = 128
) -> int:
    ratio = base_seq / max(seq_len, 1)
    return max(1, int(base_batch * ratio))


def apply_gradient_sign_consistency(
    model: nn.Module, grad_history: dict, threshold: int = 6, window: int = 8,
):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        sign = param.grad.sign()

        if name not in grad_history:
            grad_history[name] = []

        grad_history[name].append(sign.detach().clone())

        if len(grad_history[name]) > window:
            grad_history[name] = grad_history[name][-window:]

        if len(grad_history[name]) >= window:
            signs = torch.stack(grad_history[name])
            pos_count = (signs > 0).sum(dim=0)
            neg_count = (signs < 0).sum(dim=0)
            max_count = torch.maximum(pos_count, neg_count)

            inconsistent = max_count < threshold
            param.grad[inconsistent] = 0.0


@torch.no_grad()
def apply_hfott_gradient_correction(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    criterion: Callable,
    sigma: float = 0.1,
    alpha: float = 0.3,
    use_amp: bool = False,
    device: str = "cuda",
) -> float:
    uncertain_params = []
    for module in model.modules():
        if hasattr(module, 'committed_mask') and hasattr(module, 'shadow_weight'):
            uncertain_mask = ~module.committed_mask
            if uncertain_mask.any():
                uncertain_params.append((module, uncertain_mask))

    if not uncertain_params:
        return 0.0

    noise_handles = []
    noises = {}

    def make_noise_hook(mod, mask):
        def hook(m, inp):
            x = inp[0]
            noise = torch.randn_like(x) * sigma
            noises[id(mod)] = noise
            return (x + noise,) + inp[1:]
        return hook

    for module, mask in uncertain_params:
        h = module.register_forward_pre_hook(make_noise_hook(module, mask))
        noise_handles.append(h)

    try:
        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device == "cuda"):
            out_noisy = model(input_ids)
            if isinstance(out_noisy, tuple):
                out_noisy = out_noisy[0]
            loss_noisy = criterion(
                out_noisy.view(-1, out_noisy.shape[-1]),
                targets.view(-1)
            )
        loss_delta = loss_noisy.item()
    finally:
        for h in noise_handles:
            h.remove()

    for module, uncertain_mask in uncertain_params:
        if module.shadow_weight.grad is not None:
            module.shadow_weight.grad[uncertain_mask] *= (1.0 + alpha)

    print(f"[HFOTT] loss_noisy={loss_noisy.item():.4f} uncertain_params={len(uncertain_params)}")
    return loss_delta


@torch.no_grad()
def apply_sparse_shadow_weights(
    model,
    optimizer,
    margin: float = 0.1,
    decay: float = 0.0,
    step: int = 0,
    log: bool = False,
):
    total_weights = 0
    evicted_weights = 0
    param_to_state = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                param_to_state[id(p)] = optimizer.state[p]
    for module in model.modules():
        if not (
            hasattr(module, "quantize_enabled")
            and hasattr(module, "group_size")
            and hasattr(module, "threshold_coeff")
        ):
            continue
        weight = module.weight
        state = param_to_state.get(id(weight))
        if state is None or "exp_avg" not in state:
            continue
        gs = module.group_size
        tc = module.threshold_coeff
        flat = weight.data.reshape(-1)
        pad_len = (gs - flat.numel() % gs) % gs
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len))
        grouped = flat.reshape(-1, gs)
        scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = grouped / scales
        abs_normalized = normalized.abs()
        threshold = tc * abs_normalized.mean(dim=-1, keepdim=True)
        distance = (abs_normalized - threshold).abs()
        stable_mask = (distance > margin).reshape(-1)
        if pad_len > 0:
            stable_mask = stable_mask[: weight.numel()]
        stable_mask = stable_mask.reshape(weight.shape)
        n_total = weight.numel()
        n_stable = stable_mask.sum().item()
        total_weights += n_total
        evicted_weights += n_stable
        if decay == 0.0:
            state["exp_avg"][stable_mask] = 0.0
            state["exp_avg_sq"][stable_mask] = 0.0
        else:
            state["exp_avg"][stable_mask] *= decay
            state["exp_avg_sq"][stable_mask] *= decay
    if log and total_weights > 0:
        pct = 100 * evicted_weights / total_weights
        print(
            f"[SPARSE-SHADOW] step={step}: {evicted_weights}/{total_weights} "
            f"({pct:.1f}%) optimizer states evicted/decayed (margin={margin})"
        )
    return evicted_weights, total_weights


def update_caq_state(model, optimizer):
    param_to_state = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state and "exp_avg_sq" in optimizer.state[p]:
                param_to_state[id(p)] = optimizer.state[p]["exp_avg_sq"]

    updated = 0
    for module in model.modules():
        if hasattr(module, "use_caq") and module.use_caq and hasattr(module, "weight"):
            param_id = id(module.weight)
            if param_id in param_to_state:
                module._v_hat = param_to_state[param_id].detach()
                updated += 1
    return updated


def update_info_freeze_vhat(model, optimizer):
    param_to_state = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state and "exp_avg_sq" in optimizer.state[p]:
                param_to_state[id(p)] = optimizer.state[p]["exp_avg_sq"]

    updated = 0
    for module in model.modules():
        if (
            hasattr(module, "use_info_freeze")
            and module.use_info_freeze
            and hasattr(module, "weight")
        ):
            param_id = id(module.weight)
            if param_id in param_to_state:
                module._v_hat = param_to_state[param_id].detach()
                updated += 1
    return updated


def zero_frozen_optimizer_state(model, optimizer):
    param_to_state = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state and len(optimizer.state[p]) > 0:
                param_to_state[id(p)] = (p, optimizer.state[p])

    deleted_params = 0
    deleted_mb = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                continue
            pid = id(p)
            if pid not in param_to_state:
                continue
            _, state = param_to_state[pid]
            bytes_freed = sum(
                v.numel() * v.element_size()
                for v in state.values()
                if isinstance(v, torch.Tensor)
            )
            optimizer.state[p].clear()
            deleted_mb += bytes_freed / (1024 ** 2)
            deleted_params += 1
            del param_to_state[pid]

    if deleted_params > 0:
        print(
            f"[ZERO-FROZEN] AutoFreeze eviction: {deleted_params} params | "
            f"{deleted_mb:.1f} MB freed"
        )

    zeroed_params = 0
    zeroed_weights = 0
    for module in model.modules():
        if not (
            hasattr(module, "use_info_freeze")
            and module.use_info_freeze
            and hasattr(module, "_freeze_mask")
            and module._freeze_mask is not None
            and hasattr(module, "weight")
        ):
            continue
        pid = id(module.weight)
        if pid not in param_to_state:
            continue
        _, state = param_to_state[pid]
        mask = module._freeze_mask
        if not mask.any():
            continue
        if "exp_avg" in state:
            state["exp_avg"][mask] = 0.0
        if "exp_avg_sq" in state:
            state["exp_avg_sq"][mask] = 0.0
        zeroed_params += 1
        zeroed_weights += mask.sum().item()

    c2_zeroed = 0
    c2_modules = 0
    c2_total_weights = 0
    for module in model.modules():
        if not (
            getattr(module, "use_commitment_tracking", False)
            and getattr(module, "committed_mask", None) is not None
            and hasattr(module, "weight")
        ):
            continue
        c2_modules += 1
        c2_total_weights += module.committed_mask.numel()
        pid = id(module.weight)
        if pid not in param_to_state:
            continue
        _, state = param_to_state[pid]
        mask = module.committed_mask
        if not mask.any():
            continue
        if "exp_avg" in state:
            state["exp_avg"][mask] = 0.0
        if "exp_avg_sq" in state:
            state["exp_avg_sq"][mask] = 0.0
        c2_zeroed += mask.sum().item()

    if c2_modules > 0 and not hasattr(zero_frozen_optimizer_state, "_hsp_c2_logged"):
        zero_frozen_optimizer_state._hsp_c2_logged = True
        zero_frozen_optimizer_state._hsp_c2_call_count = 0
        _c2_pct = 100 * c2_zeroed / c2_total_weights if c2_total_weights > 0 else 0
        print(
            f"[HSP-C2] Active: {c2_modules} C2 modules, "
            f"{int(c2_zeroed)}/{c2_total_weights} committed ({_c2_pct:.1f}%) "
            f"~{c2_zeroed * 4 / 1e6:.1f}MB optimizer state freed"
        )
    elif c2_zeroed > 0:
        zero_frozen_optimizer_state._hsp_c2_call_count = getattr(
            zero_frozen_optimizer_state, "_hsp_c2_call_count", 0
        ) + 1
        if zero_frozen_optimizer_state._hsp_c2_call_count % 50 == 0:
            _c2_pct = 100 * c2_zeroed / c2_total_weights if c2_total_weights > 0 else 0
            print(
                f"[HSP-C2] step~{zero_frozen_optimizer_state._hsp_c2_call_count} "
                f"committed={int(c2_zeroed)}/{c2_total_weights} ({_c2_pct:.1f}%) "
                f"~{c2_zeroed * 4 / 1e6:.1f}MB freed"
            )

    return zeroed_weights + c2_zeroed


def apply_commitment_lr_scaling(model: nn.Module) -> dict:
    scales_applied = {}
    LR_LOW = 0.5
    LR_HIGH = 2.0

    for name, module in model.named_modules():
        if not hasattr(module, "threshold_coeff"):
            continue
        if not getattr(module, "use_commitment_tracking", False):
            continue
        n_stable = getattr(module, "n_stable", None)
        if n_stable is None:
            continue

        n_target = getattr(module, "commitment_n_stable_target", 100)
        c_l = (n_stable.float().mean() / max(1, n_target)).clamp(0.0, 1.0).item()
        scale = LR_HIGH + (LR_LOW - LR_HIGH) * c_l

        if module.weight.grad is not None:
            module.weight.grad.mul_(scale)
        scales_applied[name] = scale

    return scales_applied


def evict_committed_params(
    optimizer: torch.optim.Optimizer,
    commitment_monitor,
    commitment_threshold: float = 0.7,
    logger=None,
) -> int:
    if commitment_monitor is None:
        return 0

    from ..components.ternary import TernaryLinear

    evicted_count = 0
    evicted_bytes = 0

    for group_idx, param_group in enumerate(optimizer.param_groups):
        params_to_remove = []

        for param in param_group["params"]:
            if param not in optimizer.state:
                continue

            opt_state = optimizer.state[param]
            if "exp_avg" not in opt_state or "exp_avg_sq" not in opt_state:
                continue

            layer_name = None
            for name, module in enumerate(
                optimizer.param_groups[0].get("_module_refs", [])
            ):
                if param in list(module.parameters()):
                    layer_name = name
                    break

            param_bytes = param.numel() * 8
            params_to_remove.append((param, param_bytes))
            evicted_count += 1
            evicted_bytes += param_bytes

        for param, param_bytes in params_to_remove:
            param_group["params"].remove(param)

    if evicted_count > 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        msg = (
            f"[PARAM_EVICTION] Evicted {evicted_count} committed params. "
            f"Memory freed: {evicted_bytes / 1e9:.3f} GB (optimizer states only). "
            f"Threshold: {commitment_threshold:.2f}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            vram_msg = (
                f"[PARAM_EVICTION] Post-eviction VRAM: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved"
            )
            if logger:
                logger.info(vram_msg)
            else:
                print(vram_msg)

    return evicted_count
