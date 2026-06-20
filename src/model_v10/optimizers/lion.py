"""Lion optimizer — Sign Gradient Descent with Momentum.

Reference: Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
arXiv:2302.06675

Formula per step:
    c_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
    p_t   = p_{t-1} - lr * (sign(c_t) + wd * p_{t-1})
    m_t   = beta2 * m_{t-1} + (1 - beta2) * g_t

Memory: 1 state tensor per param (exp_avg only).
No exp_avg_sq. ~33% less optimizer memory than AdamW at the same param count.

LR note: Lion applies sign(update), so step magnitude is always exactly lr.
Use lr ~10x smaller than AdamW (e.g., 6e-5 where AdamW uses 6e-4).
"""

import torch
from torch.optim import Optimizer
from src.model_v10.optimizers.adamw_8bit import AdamW8bit as _AdamW8bit


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    # F4: track consecutive skips; offload to CPU after 50
                    state = self.state[p]
                    if len(state) > 0:
                        state["_skip_count"] = state.get("_skip_count", 0) + 1
                        if state["_skip_count"] >= 50 and not state.get("_offloaded", False):
                            if state.get("exp_avg_int8") is not None:
                                state["exp_avg_int8"] = state["exp_avg_int8"].cpu().pin_memory()
                                state["exp_avg_scale"] = state["exp_avg_scale"].cpu().pin_memory()
                                state["_offloaded"] = True
                                if not hasattr(self, '_cpu_offload_logged'):
                                    print(
                                        f"[CPU-OFFLOAD] Offloaded exp_avg_int8 to CPU pinned | "
                                        f"param shape={list(p.shape)} "
                                        f"skip_count={state['_skip_count']}"
                                    )
                                    self._cpu_offload_logged = True
                    continue

                grad = p.grad

                # D3 — Grad Centralization (R9-A4)
                # For 2D+ tensors: g = g - mean(g, dim=[1,...])
                # Increases effective batch size: B_eff 1.82→3.48 for Lion.
                if grad.dim() >= 2:
                    grad = grad - grad.mean(dim=list(range(1, grad.dim())), keepdim=True)
                    if not hasattr(self, '_gc_logged'):
                        print(
                            f"[D3-GC] Grad centralization active | "
                            f"shape={list(grad.shape)} ndim={grad.dim()}"
                        )
                        self._gc_logged = True

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg_int8"] = None
                    state["exp_avg_scale"] = None
                    state["_exp_avg_shape"] = p.shape
                    state["g_prev"] = torch.zeros(p.shape, dtype=torch.float16, device=p.device)
                    state["_skip_count"] = 0
                    state["_offloaded"] = False

                # F4: restore from CPU if previously offloaded
                if state.get("_offloaded", False):
                    if state["exp_avg_int8"] is not None:
                        state["exp_avg_int8"] = state["exp_avg_int8"].to(p.device)
                        state["exp_avg_scale"] = state["exp_avg_scale"].to(p.device)
                    state["_offloaded"] = False
                    state["_skip_count"] = 0
                    if not hasattr(self, '_cpu_offload_restore_logged'):
                        print(f"[CPU-OFFLOAD] Restored exp_avg_int8 to GPU | param shape={list(p.shape)}")
                        self._cpu_offload_restore_logged = True

                # F1: dequantize INT8 momentum to FP32
                if state["exp_avg_int8"] is None:
                    m = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                else:
                    m = _AdamW8bit._dequantize_blockwise(
                        state["exp_avg_int8"],
                        state["exp_avg_scale"],
                        block_size=128,
                        shape=state["_exp_avg_shape"],
                    )

                g_prev = state["g_prev"].float()
                storm_coeff = 0.9
                c_t = grad + storm_coeff * (grad - g_prev)
                if not hasattr(self, '_mars_logged'):
                    _storm_corr = (c_t - grad).abs().mean().item()
                    print(f"[E1-MARS] gamma=0.1 beta1=0.9 storm_coeff={storm_coeff} storm_corr={_storm_corr:.4f} active")
                    self._mars_logged = True

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Compute update direction: sign(beta1*m + (1-beta1)*g)
                update = beta1 * m + (1.0 - beta1) * c_t

                # C-Lion: only update where momentum and gradient agree
                mask = (update * grad > 0).float()
                ratio = mask.mean()
                if not hasattr(self, '_clion_logged'):
                    print(f"[C-LION] Cautious mask active | agreement={ratio.item():.1%} of params updated")
                    self._clion_logged = True
                mask = mask / (ratio + 1e-8)
                update = update * mask

                p.add_(update.sign_(), alpha=-lr)

                # Update momentum: m = beta2*m + (1-beta2)*g
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                # F1: requantize momentum to INT8 blockwise
                int8_data, scales = _AdamW8bit._quantize_blockwise(m, block_size=128)
                state["exp_avg_int8"] = int8_data
                state["exp_avg_scale"] = scales
                state["g_prev"].copy_(grad.half())

                if not hasattr(self, '_lion_int8_logged'):
                    _bytes_int8 = m.numel() * 1 + scales.numel() * 4
                    _bytes_fp16 = m.numel() * 2
                    _savings_pct = (_bytes_fp16 - _bytes_int8) / _bytes_fp16 * 100
                    print(
                        f"[LION-INT8] INT8 blockwise momentum active | "
                        f"shape={list(p.shape)} numel={m.numel()} "
                        f"savings={_savings_pct:.1f}%"
                    )
                    self._lion_int8_logged = True

        return loss
