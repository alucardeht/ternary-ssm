"""TernarySignAccumulator (TSA) — INT8 counter optimizer for ternary weights.

Replaces Adam/LION for ternary params entirely.
- No shadow weight update (direct ternary flip)
- 1 byte/param vs Adam's 8 bytes/param = 8x memory reduction
- No momentum, no variance: just sign(grad) accumulation

Algorithm per step:
    accum += clip(sign(grad), -127, 127)   # or += sign(grad) directly
    if |accum| >= threshold:
        w_ternary = -w_ternary if w_ternary != 0 else sign(accum)
        accum = 0

Non-ternary params fall back to AdamW (unchanged).
"""

import torch
from torch.optim import Optimizer


class TernarySignAccumulator(Optimizer):
    """INT8 counter optimizer for ternary weights {-1, 0, +1}.

    Param groups must carry 'is_ternary': True for ternary params.

    Args:
        param_groups: from model.get_aga_param_split()
        tsa_threshold: int, |accumulator| threshold to trigger flip (default 50)
        lr: learning rate for non-ternary (AdamW) params
        weight_decay: for non-ternary params
        betas: Adam betas for non-ternary params
        eps: Adam eps for non-ternary params
    """

    def __init__(
        self,
        param_groups,
        tsa_threshold: int = 50,
        threshold_final: int = 12,
        total_steps: int = 10000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        cpu_offload: bool = False,
        sparse_epsilon: float = 0.5,
    ):
        defaults = dict(
            tsa_threshold=tsa_threshold,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(param_groups, defaults)
        self.total_steps = 0
        self.total_flips = 0
        self.threshold_final = threshold_final
        self.total_steps_schedule = total_steps
        self._global_step = 0
        self.cpu_offload = cpu_offload
        self.sparse_epsilon = sparse_epsilon
        self._surprise_w = 1
        self._ema_surprise = 1.0

        if cpu_offload:
            self._buf_A = {}
            self._buf_B = {}
            self._cpu_accums = {}
            self._compute_stream = torch.cuda.Stream()
            self._transfer_stream = torch.cuda.Stream()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_steps += 1
        self._global_step += 1

        if self.cpu_offload:
            self._step_ternary_offloaded()
        else:
            for group in self.param_groups:
                if group.get('is_ternary', False):
                    self._step_ternary(group)

        for group in self.param_groups:
            if not group.get('is_ternary', False):
                self._step_adamw(group)

        return loss

    def enable_cpu_offload(self, ternary_by_layer):
        """Initialize CPU offload infrastructure for TSA accumulators.

        Args:
            ternary_by_layer: list of param lists, one per layer (from model.get_aga_param_split_layered())
        """
        if not self.cpu_offload:
            return

        for layer_idx, params in enumerate(ternary_by_layer):
            for p in params:
                if not p.requires_grad:
                    continue
                pid = id(p)
                cpu_acc = torch.zeros(
                    p.data.shape,
                    dtype=torch.int8,
                    pin_memory=True,
                    device='cpu'
                )
                self._cpu_accums[pid] = cpu_acc
                self._buf_A[pid] = torch.zeros_like(p.data, dtype=torch.int8)
                self._buf_B[pid] = torch.zeros_like(p.data, dtype=torch.int8)
                state = self.state[p]
                if len(state) == 0:
                    state['accum'] = cpu_acc

    def set_surprise_weight(self, w: float):
        """Call from training loop after loss, before optimizer.step(). w in [1.0, 16.0]."""
        self._surprise_w = max(1, min(16, int(round(w))))
        self._ema_surprise = 0.99 * self._ema_surprise + 0.01 * self._surprise_w

    def _step_ternary(self, group):
        base_threshold = group['tsa_threshold']
        sw = self._surprise_w

        ema_w = max(1.0, self._ema_surprise)
        base_threshold_eff = max(4, round(base_threshold / ema_w))
        threshold_final_eff = max(2, round(self.threshold_final / ema_w))

        progress = min(self._global_step / self.total_steps_schedule, 1.0)
        if progress > 0.5:
            anneal = (progress - 0.5) * 2
            threshold = int(base_threshold_eff * (1 - anneal) + threshold_final_eff * anneal)
        else:
            threshold = base_threshold_eff

        warmup_steps = 2 * base_threshold
        in_warmup = self._global_step < warmup_steps

        for p in group['params']:
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                continue

            state = self.state[p]
            if len(state) == 0:
                # INT8 accumulator — 1 byte/param
                state['accum'] = torch.zeros_like(p.data, dtype=torch.int8)

            accum = state['accum']

            # Accumulate sign of gradient (clipped to INT8 range implicitly by sign)
            # sign() returns {-1, 0, 1} as float — cast to int8
            g_sign = p.grad.sign().to(torch.int8)

            if in_warmup:
                accum.add_(g_sign * sw)
            else:
                epsilon = int(threshold * self.sparse_epsilon)
                near_flip_mask = accum.abs() >= (threshold - epsilon)
                masked_vote = (g_sign * sw) * near_flip_mask.to(torch.int8)
                accum.add_(masked_vote)

            # Clamp to avoid INT8 overflow (shouldn't happen at normal threshold values)
            accum.clamp_(-127, 127)

            # Identify params that should flip
            flip_mask = accum.abs() >= threshold

            if flip_mask.any():
                # Direct ternary flip — no shadow weight needed
                # w is already {-1, 0, 1} (or close to it after STE)
                flip_sign = accum[flip_mask].sign().float()

                # If w == 0: set to sign(accum)
                # If w != 0: flip sign
                w_flat = p.data[flip_mask]
                is_zero = w_flat.abs() < 0.5
                w_flat[is_zero] = flip_sign[is_zero]
                w_flat[~is_zero] = -w_flat[~is_zero]
                p.data[flip_mask] = w_flat

                # Reset accumulator for flipped params
                accum[flip_mask] = 0

                n_flips = flip_mask.sum().item()
                self.total_flips += n_flips

            # Activation log (verification)
            if self.total_steps % 50 == 1 and not torch.compiler.is_compiling():
                if not in_warmup:
                    epsilon = int(threshold * self.sparse_epsilon)
                    near_flip_mask = accum.abs() >= (threshold - epsilon)
                    near_frac = near_flip_mask.float().mean().item()
                else:
                    near_frac = 1.0
                warmup_str = " (warmup)" if in_warmup else ""
                print(
                    f"[TSA] step={self.total_steps} "
                    f"flips_this_period={self.total_flips} "
                    f"avg_accum={accum.float().abs().mean():.2f} "
                    f"threshold={threshold} "
                    f"near_flip_frac={near_frac:.3f} "
                    f"surprise_w={self._surprise_w} ema_surprise={self._ema_surprise:.2f}"
                    f"{warmup_str}"
                )

    def _step_ternary_group_inplace(self, group, buf):
        """Run TSA update for one param group using provided GPU buffer tensors.

        buf: dict mapping id(p) -> GPU INT8 tensor (current accumulator on GPU)
        Modifies buf[id(p)] and p.data in-place.
        """
        base_threshold = group['tsa_threshold']
        sw = self._surprise_w

        ema_w = max(1.0, self._ema_surprise)
        base_threshold_eff = max(4, round(base_threshold / ema_w))
        threshold_final_eff = max(2, round(self.threshold_final / ema_w))

        progress = min(self._global_step / self.total_steps_schedule, 1.0)
        if progress > 0.5:
            anneal = (progress - 0.5) * 2
            threshold = int(base_threshold_eff * (1 - anneal) + threshold_final_eff * anneal)
        else:
            threshold = base_threshold_eff

        warmup_steps = 2 * base_threshold
        in_warmup = self._global_step < warmup_steps

        for p in group['params']:
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                continue

            pid = id(p)
            accum = buf[pid]

            g_sign = p.grad.sign().to(torch.int8)

            if in_warmup:
                accum.add_(g_sign * sw)
            else:
                epsilon = int(threshold * self.sparse_epsilon)
                near_flip_mask = accum.abs() >= (threshold - epsilon)
                masked_vote = (g_sign * sw) * near_flip_mask.to(torch.int8)
                accum.add_(masked_vote)

            accum.clamp_(-127, 127)

            flip_mask = accum.abs() >= threshold
            if flip_mask.any():
                flip_sign = accum[flip_mask].sign().float()
                w_flat = p.data[flip_mask]
                is_zero = w_flat.abs() < 0.5
                w_flat[is_zero] = flip_sign[is_zero]
                w_flat[~is_zero] = -w_flat[~is_zero]
                p.data[flip_mask] = w_flat
                accum[flip_mask] = 0
                self.total_flips += flip_mask.sum().item()

        if self.total_steps % 50 == 1 and not torch.compiler.is_compiling():
            for p in group['params']:
                if p.grad is not None:
                    pid = id(p)
                    accum = buf[pid]
                    if not in_warmup:
                        epsilon = int(threshold * self.sparse_epsilon)
                        near_flip_mask = accum.abs() >= (threshold - epsilon)
                        near_frac = near_flip_mask.float().mean().item()
                    else:
                        near_frac = 1.0
                    layer_idx = group.get('layer_idx', '?')
                    warmup_str = " (warmup)" if in_warmup else ""
                    print(
                        f"[TSA-OFFLOAD] step={self.total_steps} layer={layer_idx} "
                        f"flips_total={self.total_flips} "
                        f"avg_accum={accum.float().abs().mean():.2f} "
                        f"threshold={threshold} "
                        f"near_flip_frac={near_frac:.3f} "
                        f"surprise_w={self._surprise_w} ema_surprise={self._ema_surprise:.2f}"
                        f"{warmup_str}"
                    )
                    break

    def _step_ternary_offloaded(self):
        """Process all ternary layers sequentially with async CPU<->GPU double-buffering.

        Buffer layout at any point during processing layer k:
            buf_A[p] = GPU tensor with acc for layer k  (being actively used)
            buf_B[p] = GPU tensor with acc for layer k+1 (prefetching while k runs)
        """
        ternary_groups = [g for g in self.param_groups if g.get('is_ternary', False)]
        ternary_groups = sorted(ternary_groups, key=lambda g: g.get('layer_idx', 0))

        if not ternary_groups:
            return

        for group in ternary_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                pid = id(p)
                state = self.state[p]
                if len(state) == 0 or pid not in self._cpu_accums:
                    cpu_acc = torch.zeros(
                        p.data.shape,
                        dtype=torch.int8,
                        pin_memory=True,
                        device='cpu'
                    )
                    self._cpu_accums[pid] = cpu_acc
                    self._buf_A[pid] = torch.zeros_like(p.data, dtype=torch.int8)
                    self._buf_B[pid] = torch.zeros_like(p.data, dtype=torch.int8)
                    state['accum'] = cpu_acc

        first_group = ternary_groups[0]
        with torch.cuda.stream(self._transfer_stream):
            for p in first_group['params']:
                if p.grad is None:
                    continue
                pid = id(p)
                self._buf_A[pid].copy_(self._cpu_accums[pid], non_blocking=True)

        for k, group in enumerate(ternary_groups):
            self._compute_stream.wait_stream(self._transfer_stream)

            if k + 1 < len(ternary_groups):
                next_group = ternary_groups[k + 1]
                with torch.cuda.stream(self._transfer_stream):
                    for p in next_group['params']:
                        if p.grad is None:
                            continue
                        pid = id(p)
                        self._buf_B[pid].copy_(self._cpu_accums[pid], non_blocking=True)

            with torch.cuda.stream(self._compute_stream):
                self._step_ternary_group_inplace(group, buf=self._buf_A)

            with torch.cuda.stream(self._transfer_stream):
                self._transfer_stream.wait_stream(self._compute_stream)
                for p in group['params']:
                    if p.grad is None:
                        continue
                    pid = id(p)
                    self._cpu_accums[pid].copy_(self._buf_A[pid], non_blocking=True)

            self._buf_A, self._buf_B = self._buf_B, self._buf_A

        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        torch.cuda.current_stream().wait_stream(self._compute_stream)

    def _step_adamw(self, group):
        """Standard AdamW for non-ternary params (embeddings, norms, lm_head)."""
        lr = group['lr']
        wd = group.get('weight_decay', 0.0)
        b1, b2 = group.get('betas', (0.9, 0.999))
        eps = group.get('eps', 1e-8)

        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

            state['step'] += 1
            step = state['step']
            m, v = state['exp_avg'], state['exp_avg_sq']
            grad = p.grad.float()

            if wd > 0:
                grad = grad.add(p.data.float(), alpha=wd)

            m.mul_(b1).add_(grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

            bias1 = 1 - b1 ** step
            bias2 = 1 - b2 ** step
            step_size = lr * (bias2 ** 0.5) / bias1

            p.data.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)

    def get_hot_mask(self, param, margin=10):
        """Return boolean mask of output rows near flip threshold.

        Hot = |accum| >= (threshold - margin). These rows are candidates
        for flipping soon and warrant sparse grad_weight computation.

        Args:
            param: the tensor to query (must be a ternary param tracked by this optimizer)
            margin: how many counts below threshold to consider "hot" (default 10)

        Returns:
            None if param not in state, else boolean tensor same shape as param
        """
        state = self.state.get(param, {})
        accum = state.get('accum', None)
        if accum is None:
            return None

        threshold = self.defaults.get('tsa_threshold', 50)
        hot = accum.abs() >= (threshold - margin)
        return hot

    def get_stats(self):
        total_ternary = 0
        for group in self.param_groups:
            if group.get('is_ternary', False):
                for p in group['params']:
                    total_ternary += p.numel()
        return {
            'total_steps': self.total_steps,
            'total_flips': self.total_flips,
            'flips_per_step': self.total_flips / max(1, self.total_steps),
            'total_ternary_params': total_ternary,
        }
