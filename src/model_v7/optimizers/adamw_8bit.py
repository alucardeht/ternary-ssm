import torch
from torch.optim import Optimizer
import math


class AdamW8bit(Optimizer):
    """8-bit AdamW with block-wise quantization of optimizer states.

    Stores exp_avg and exp_avg_sq as INT8 with per-block FP32 scales.
    Reduces optimizer state memory by ~4x while maintaining numerical stability.

    Standard AdamW math:
    - m = beta1 * m + (1 - beta1) * grad
    - v = beta2 * v + (1 - beta2) * grad^2
    - update = m / (sqrt(v) + eps), bias-corrected

    Quantization:
    - Split m, v into blocks (block_size=128)
    - For each block: scale = max(|block|) / 127, int8 = stochastic_round(values/scale)
    - Dequantize to FP32 before update, requantize after

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        betas: Adam betas (default: (0.9, 0.999))
        eps: Epsilon for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        block_size: Block size for quantization (default: 128)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, block_size=128):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    @staticmethod
    def _quantize_blockwise(tensor_fp32, block_size=128):
        """Quantize tensor to INT8 with per-block FP32 scales using stochastic rounding.

        Args:
            tensor_fp32: FP32 tensor to quantize
            block_size: Size of quantization blocks

        Returns:
            (int8_data, scales) tuple
        """
        flat = tensor_fp32.reshape(-1)
        n = flat.numel()
        num_blocks = (n + block_size - 1) // block_size

        if n % block_size != 0:
            pad_size = num_blocks * block_size - n
            flat_padded = torch.nn.functional.pad(flat, (0, pad_size), value=0.0)
        else:
            flat_padded = flat

        blocks = flat_padded.reshape(num_blocks, block_size)
        abs_max = blocks.abs().amax(dim=1, keepdim=True)
        scales = (abs_max / 127.0).clamp(min=1e-12)

        continuous = blocks / scales
        floored = continuous.floor()
        frac = continuous - floored
        rand = torch.rand_like(frac)
        quantized = (floored + (rand < frac).float()).clamp(-127, 127)

        int8_data = quantized.reshape(-1)[:n].to(torch.int8)
        scales_flat = scales.squeeze(1)
        return int8_data, scales_flat

    @staticmethod
    def _dequantize_blockwise(int8_data, scales, block_size=128, shape=None):
        """Dequantize INT8 tensor with per-block scales to FP32.

        Args:
            int8_data: INT8 quantized data (flat)
            scales: FP32 per-block scales
            block_size: Block size used in quantization
            shape: Target shape for output tensor

        Returns:
            FP32 dequantized tensor
        """
        flat = int8_data.float()
        n = flat.numel()
        num_blocks = scales.numel()

        if n % block_size != 0:
            pad_size = num_blocks * block_size - n
            flat_padded = torch.nn.functional.pad(flat, (0, pad_size), value=0.0)
        else:
            flat_padded = flat

        blocks = flat_padded.reshape(num_blocks, block_size)
        blocks = blocks * scales.unsqueeze(1)
        flat = blocks.reshape(-1)[:n]

        if shape is not None:
            flat = flat.reshape(shape)

        return flat

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimizer step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']
            block_size = group['block_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW8bit does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_int8'] = None
                    state['exp_avg_scale'] = None
                    state['exp_avg_sq_int8'] = None
                    state['exp_avg_sq_scale'] = None

                state['step'] += 1
                step = state['step']

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                if state['exp_avg_int8'] is None:
                    exp_avg = torch.zeros_like(p, dtype=torch.float32)
                    exp_avg_sq = torch.zeros_like(p, dtype=torch.float32)
                else:
                    exp_avg = self._dequantize_blockwise(
                        state['exp_avg_int8'],
                        state['exp_avg_scale'],
                        block_size=block_size,
                        shape=p.shape
                    )
                    exp_avg_sq = self._dequantize_blockwise(
                        state['exp_avg_sq_int8'],
                        state['exp_avg_sq_scale'],
                        block_size=block_size,
                        shape=p.shape
                    )

                grad_fp32 = grad.float()
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                p.data.add_(exp_avg / denom, alpha=-step_size)

                exp_avg_int8, exp_avg_scale = self._quantize_blockwise(exp_avg, block_size)
                exp_avg_sq_int8, exp_avg_sq_scale = self._quantize_blockwise(exp_avg_sq, block_size)

                state['exp_avg_int8'] = exp_avg_int8
                state['exp_avg_scale'] = exp_avg_scale
                state['exp_avg_sq_int8'] = exp_avg_sq_int8
                state['exp_avg_sq_scale'] = exp_avg_sq_scale

        return loss
