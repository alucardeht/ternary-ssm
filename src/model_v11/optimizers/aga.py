"""AGA: Algebraic Gradient Accumulation optimizer for ternary layers.

Replaces AdamW's momentum (m) and variance (v) states with a single
INT16 sign accumulator per weight. When the accumulator crosses threshold K,
applies a calibrated step to the FP32 shadow weight.

Memory: 2 bytes/weight (INT16 accum) vs 8 bytes/weight (FP32 m + FP32 v)
"""

import torch
from torch.optim import Optimizer


class AGAOptimizer(Optimizer):
    """Algebraic Gradient Accumulation for ternary weight layers.

    Args:
        params: Ternary weight parameters (FP32 shadow weights)
        lr: Step size applied when accumulator fires (default: 1e-3)
        threshold_K: Required accumulated sign count before update (default: 32)
        decay_steps: Every N steps, halve accumulators to prevent staleness (0=disable)
        ternary_pull: Pull shadow weights toward nearest ternary value (0=disable)
        use_int16: Use INT16 accumulators (default: True)
    """

    def __init__(self, params, lr=1e-3, threshold_K=32, decay_steps=0,
                 ternary_pull=0.0, use_int16=True):
        if threshold_K < 1:
            raise ValueError(f"threshold_K must be >= 1, got {threshold_K}")
        defaults = dict(lr=lr, threshold_K=threshold_K, decay_steps=decay_steps,
                        ternary_pull=ternary_pull, use_int16=use_int16)
        super().__init__(params, defaults)
        self.total_flips = 0
        self.total_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step_flips = 0

        for group in self.param_groups:
            lr = group['lr']
            K = group['threshold_K']
            decay_steps = group['decay_steps']
            pull = group['ternary_pull']
            use_int16 = group['use_int16']

            for p in group['params']:
                if p.grad is None:
                    continue

                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    continue

                state = self.state[p]

                if len(state) == 0:
                    acc_dtype = torch.int16 if use_int16 else torch.float32
                    state['accumulator'] = torch.zeros_like(p.data, dtype=acc_dtype)
                    state['step'] = 0
                    state['flip_count'] = 0

                acc = state['accumulator']
                state['step'] += 1

                if decay_steps > 0 and state['step'] % decay_steps == 0:
                    if use_int16:
                        acc.div_(2)
                    else:
                        acc.mul_(0.5)

                grad_sign = p.grad.data.sign()
                if use_int16:
                    acc.add_(grad_sign.to(torch.int16))
                else:
                    acc.add_(grad_sign)

                pos_fire = acc >= K
                neg_fire = acc <= -K

                if pos_fire.any() or neg_fire.any():
                    n_flips = pos_fire.sum().item() + neg_fire.sum().item()
                    state['flip_count'] += n_flips
                    step_flips += n_flips

                    p.data[pos_fire] -= lr
                    p.data[neg_fire] += lr

                    acc[pos_fire] = 0
                    acc[neg_fire] = 0

                if pull > 0:
                    nearest = p.data.sign() * (p.data.abs() > 0.35).float()
                    diff = nearest - p.data
                    p.data.add_(diff, alpha=pull)

        self.total_flips += step_flips
        self.total_steps += 1
        return loss

    def get_stats(self):
        total_params = 0
        total_acc_abs = 0.0
        max_acc = 0
        near_threshold = 0

        for group in self.param_groups:
            K = group['threshold_K']
            for p in group['params']:
                if p not in self.state:
                    continue
                state = self.state[p]
                acc = state['accumulator']
                n = acc.numel()
                total_params += n
                abs_acc = acc.abs().float()
                total_acc_abs += abs_acc.sum().item()
                cur_max = abs_acc.max().item()
                if cur_max > max_acc:
                    max_acc = cur_max
                near_threshold += (abs_acc >= K * 0.8).sum().item()

        return {
            'total_steps': self.total_steps,
            'total_flips': self.total_flips,
            'flips_per_step': self.total_flips / max(1, self.total_steps),
            'avg_abs_accumulator': total_acc_abs / max(1, total_params),
            'max_accumulator': max_acc,
            'near_threshold_pct': near_threshold / max(1, total_params) * 100,
            'total_params': total_params,
        }


class FusedAGAAccumulator:
    """Stateful accumulator for fused gradient cascade with AGA.

    Instead of the standard flow: backward -> materialize all grads -> optimizer.step()
    This streams gradients directly into INT16 accumulators via hooks, zeroing each
    gradient immediately after consumption.

    R2-A2-M4: Eliminates peak gradient memory for ternary params.
    """

    def __init__(self, lr=1e-3, threshold_K=32, decay_steps=0,
                 ternary_pull=0.0, use_int16=True):
        self.lr = lr
        self.threshold_K = threshold_K
        self.decay_steps = decay_steps
        self.ternary_pull = ternary_pull
        self.use_int16 = use_int16

        self.accumulators = {}  # param_id -> INT16 tensor
        self.step_counts = {}   # param_id -> int
        self.flip_counts = {}   # param_id -> int
        self.total_flips = 0
        self.total_steps = 0
        self._hooks = []
        self._enabled = True  # Can be temporarily disabled (e.g., during eval)
        self._step_flips = 0  # Flips accumulated in current optimizer step

    def _get_or_create_state(self, param):
        pid = id(param)
        if pid not in self.accumulators:
            acc_dtype = torch.int16 if self.use_int16 else torch.float32
            self.accumulators[pid] = torch.zeros_like(param.data, dtype=acc_dtype)
            self.step_counts[pid] = 0
            self.flip_counts[pid] = 0
        return self.accumulators[pid]

    @torch.no_grad()
    def _hook_fn(self, param):
        """Post-accumulate-grad hook: consume gradient immediately, update accumulator."""
        if not self._enabled:
            return
        if param.grad is None:
            return
        if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
            param.grad.zero_()  # Zero bad grads immediately
            return

        acc = self._get_or_create_state(param)

        # Extract sign and accumulate
        grad_sign = param.grad.data.sign()
        if self.use_int16:
            acc.add_(grad_sign.to(torch.int16))
        else:
            acc.add_(grad_sign)

        # Check threshold and apply updates
        pos_fire = acc >= self.threshold_K
        neg_fire = acc <= -self.threshold_K

        if pos_fire.any() or neg_fire.any():
            n_flips = pos_fire.sum().item() + neg_fire.sum().item()
            pid = id(param)
            self.flip_counts[pid] = self.flip_counts.get(pid, 0) + n_flips
            self._step_flips += n_flips

            param.data[pos_fire] -= self.lr
            param.data[neg_fire] += self.lr

            acc[pos_fire] = 0
            acc[neg_fire] = 0

        # Ternary pull (optional)
        if self.ternary_pull > 0:
            nearest = param.data.sign() * (param.data.abs() > 0.35).float()
            diff = nearest - param.data
            param.data.add_(diff, alpha=self.ternary_pull)

        # CRITICAL: Zero the gradient immediately to free memory for reuse
        param.grad.zero_()

    def install_hooks(self, ternary_params):
        """Register post_accumulate_grad hooks on all ternary parameters.

        Args:
            ternary_params: list of nn.Parameter (the FP32 shadow weights of TernaryLinear)

        Returns:
            Number of hooks installed
        """
        for p in ternary_params:
            # post_accumulate_grad_hook fires after gradient is fully accumulated
            # (including gradient accumulation from multiple micro-batches within
            #  the same backward call)
            handle = p.register_post_accumulate_grad_hook(
                lambda param, self=self: self._hook_fn(param)
            )
            self._hooks.append(handle)
            # Pre-create state to avoid allocation during backward
            self._get_or_create_state(p)
        return len(self._hooks)

    def begin_step(self):
        """Call at start of optimizer step cycle. Handles decay."""
        self._step_flips = 0
        self.total_steps += 1

        if self.decay_steps > 0 and self.total_steps % self.decay_steps == 0:
            for pid, acc in self.accumulators.items():
                if self.use_int16:
                    acc.div_(2)
                else:
                    acc.mul_(0.5)

    def end_step(self):
        """Call at end of optimizer step cycle. Collects stats."""
        self.total_flips += self._step_flips

    def remove_hooks(self):
        """Remove all gradient hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_stats(self):
        """Return diagnostic stats (same interface as AGAOptimizer.get_stats)."""
        total_params = 0
        total_acc_abs = 0.0
        max_acc = 0
        near_threshold = 0

        for pid, acc in self.accumulators.items():
            n = acc.numel()
            total_params += n
            abs_acc = acc.abs().float()
            total_acc_abs += abs_acc.sum().item()
            cur_max = abs_acc.max().item()
            if cur_max > max_acc:
                max_acc = cur_max
            near_threshold += (abs_acc >= self.threshold_K * 0.8).sum().item()

        return {
            'total_steps': self.total_steps,
            'total_flips': self.total_flips,
            'flips_per_step': self.total_flips / max(1, self.total_steps),
            'avg_abs_accumulator': total_acc_abs / max(1, total_params),
            'max_accumulator': max_acc,
            'near_threshold_pct': near_threshold / max(1, total_params) * 100,
            'total_params': total_params,
        }
