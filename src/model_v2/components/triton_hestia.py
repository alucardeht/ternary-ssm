import math
import torch
import triton
import triton.language as tl


BLOCK_SIZE = 128


@triton.jit
def hestia_forward_kernel(
    weights_ptr,
    gamma_ptr,
    variance_ptr,
    w_soft_ptr,
    num_elements,
    group_size,
    tau,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    start_idx = group_idx * group_size

    offsets = tl.arange(0, BLOCK_SIZE)
    elem_idx = start_idx + offsets

    mask = elem_idx < num_elements

    w_vals = tl.load(weights_ptr + elem_idx, mask=mask, other=0.0)

    w_abs = tl.abs(w_vals)
    w_sum = tl.sum(w_abs)
    gamma = w_sum / group_size
    gamma = tl.maximum(gamma, 1e-8)

    w_scaled = w_vals / gamma

    tau_safe = tl.maximum(tau, 0.01)

    dist_neg_sq = (w_scaled + 1.0) * (w_scaled + 1.0)
    dist_zero_sq = w_scaled * w_scaled
    dist_pos_sq = (w_scaled - 1.0) * (w_scaled - 1.0)

    logit_neg = -dist_neg_sq / tau_safe
    logit_zero = -dist_zero_sq / tau_safe
    logit_pos = -dist_pos_sq / tau_safe

    logit_max = tl.maximum(tl.maximum(logit_neg, logit_zero), logit_pos)

    logit_neg_stable = logit_neg - logit_max
    logit_zero_stable = logit_zero - logit_max
    logit_pos_stable = logit_pos - logit_max

    exp_neg = tl.exp(logit_neg_stable)
    exp_zero = tl.exp(logit_zero_stable)
    exp_pos = tl.exp(logit_pos_stable)

    exp_sum = exp_neg + exp_zero + exp_pos

    p_neg = exp_neg / exp_sum
    p_zero = exp_zero / exp_sum
    p_pos = exp_pos / exp_sum

    w_soft = gamma * (p_pos - p_neg)

    mean_q = p_pos - p_neg
    variance = (1.0 - p_zero) - mean_q * mean_q

    tl.store(gamma_ptr + group_idx, gamma)
    tl.store(w_soft_ptr + elem_idx, w_soft, mask=mask)
    tl.store(variance_ptr + elem_idx, variance, mask=mask)


@triton.jit
def hestia_backward_kernel(
    grad_output_ptr,
    variance_ptr,
    grad_input_ptr,
    num_elements,
    tau,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_elements

    grad_out = tl.load(grad_output_ptr + idx, mask=mask, other=0.0)
    variance = tl.load(variance_ptr + idx, mask=mask, other=0.0)

    tau_safe = tl.maximum(tau, 0.01)
    jacobian = (2.0 / tau_safe) * variance

    grad_in = grad_out * jacobian

    tl.store(grad_input_ptr + idx, grad_in, mask=mask)


class HestiaTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights: torch.Tensor, group_size: int = 128, tau: float = 1.0) -> torch.Tensor:
        orig_shape = weights.shape
        flat = weights.reshape(-1)
        num_elements = flat.numel()

        pad_len = (group_size - num_elements % group_size) % group_size
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len))

        num_groups = (flat.numel() + group_size - 1) // group_size

        gamma = torch.zeros(num_groups, device=weights.device, dtype=weights.dtype)
        variance = torch.zeros_like(flat)
        w_soft = torch.zeros_like(flat)

        grid = (num_groups,)
        hestia_forward_kernel[grid](
            flat,
            gamma,
            variance,
            w_soft,
            num_elements,
            group_size,
            tau,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        result = w_soft[:num_elements].reshape(orig_shape)
        var_result = variance[:num_elements].reshape(orig_shape)

        ctx.save_for_backward(var_result)
        ctx.tau = tau

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        var, = ctx.saved_tensors
        tau = ctx.tau

        flat_grad = grad_output.reshape(-1)
        flat_var = var.reshape(-1)
        num_elements = flat_grad.numel()

        grad_input = torch.zeros_like(flat_grad)

        grid = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        hestia_backward_kernel[grid](
            flat_grad,
            flat_var,
            grad_input,
            num_elements,
            tau,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_input.reshape(grad_output.shape), None, None


def hestia_quantize_triton(
    weight: torch.Tensor,
    group_size: int = 128,
    tau: float = 1.0,
) -> torch.Tensor:
    return HestiaTritonFunction.apply(weight, group_size, tau)


def enable_triton_hestia(model: torch.nn.Module) -> None:
    from src.model_v2.components.ternary import TernaryLinear

    for module in model.modules():
        if isinstance(module, TernaryLinear):
            module.use_triton_hestia = True
