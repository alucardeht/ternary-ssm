"""TernarySignOptimizer — ternary + FP16 hybrid optimizer with LION dual momentum.

Replaces AdamW entirely. Dual-mode:
- Ternary params: LION dual momentum (β1=0.9, β2=0.99) with FP32 momentum buffer (4 bytes/param)
- FP16 params: SGD with momentum + weight decay

Built-in progressive weight freezing based on ternary value stability (counts steps at same value).

Memory: ~6 bytes/param total (momentum 4 + frozen 1 bit + stability 2) vs 12 bytes for AdamW.
"""

import torch
from torch.optim import Optimizer


class TernarySignOptimizer(Optimizer):
    """LION optimizer for ternary + FP16 hybrid training.

    Handles two parameter types via param_groups with 'is_ternary' flag:

    Ternary (is_ternary=True):
    - FP32 momentum accumulates gradient signs (dual momentum: β1=0.9, β2=0.99)
    - Updates shadow weights EVERY step: p -= lr * sign(0.9*m + 0.1*g)
    - Momentum evolves: m = 0.99*m + 0.01*g_sign (β2 accumulation)
    - Tracks ternary value stability (consecutive steps at same value)
    - Optionally freezes weights that reach stability threshold

    FP16 (is_ternary=False):
    - SGD with momentum and weight decay
    - No special quantization

    Args:
        param_groups: List of param group dicts. Each must have 'is_ternary' flag.
        lr: Learning rate for FP16 params (default: 1e-3)
        ternary_lr: Step size for ternary shadow weight updates (default: 1e-3)
        momentum: Momentum coefficient for FP16 params (default: 0.9)
        weight_decay: L2 regularization for FP16 params (default: 0.0)
        commitment_threshold_steps: Steps to count before freezing (default: 200)
        freeze_enabled: Enable progressive weight freezing (default: False)
    """

    def __init__(
        self,
        param_groups,
        lr=1e-3,
        ternary_lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        commitment_threshold_steps=200,
        freeze_enabled=False,
    ):
        defaults = dict(
            lr=lr,
            ternary_lr=ternary_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            commitment_threshold_steps=commitment_threshold_steps,
            freeze_enabled=freeze_enabled,
        )
        super().__init__(param_groups, defaults)

        self.total_steps = 0
        self.total_flips = 0
        self.total_frozen_params = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimizer step.

        For ternary params:
        - Update FP32 momentum: m = 0.99*m + 0.01*g_sign (β2=0.99)
        - Compute mixed update: mixed = 0.9*m + 0.1*g_sign (β1=0.9)
        - Update shadow weights EVERY step using sign(mixed)
        - Track ternary value stability
        - Freeze weights that reach stability threshold

        For FP16 params:
        - SGD with momentum and weight decay
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_steps += 1

        for group in self.param_groups:
            is_ternary = group.get('is_ternary', False)

            if is_ternary:
                self._step_ternary_params(group)
            else:
                self._step_fp16_params(group)

        return loss

    def _step_ternary_params(self, group):
        """Update ternary parameters using LION dual momentum every step.

        FP32 momentum acts as EMA buffer of gradient signs (β2=0.99).
        Update computed as sign(0.9*m + 0.1*g_sign) for immediate responsiveness (β1=0.9).
        Shadow weights updated every step by ±ternary_lr based on mixed sign.
        This is full LION algorithm optimized for ternary weights.
        """
        ternary_lr = group['ternary_lr']
        freeze_enabled = group['freeze_enabled']

        for p in group['params']:
            if p.grad is None:
                continue
            if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                continue

            state = self.state[p]

            if len(state) == 0:
                state['momentum'] = torch.zeros_like(
                    p.data, dtype=torch.float32, device=p.device
                )
                state['step'] = 0
                state['frozen'] = torch.zeros_like(
                    p.data, dtype=torch.bool, device=p.device
                )
                state['prev_ternary'] = torch.zeros_like(
                    p.data, dtype=torch.float32, device=p.device
                )
                state['stability'] = torch.zeros_like(
                    p.data, dtype=torch.int16, device=p.device
                )
                state['flip_count'] = 0
                threshold = 0.7 * p.data.abs().mean()
                state['prev_ternary'][p.data > threshold] = 1
                state['prev_ternary'][p.data < -threshold] = -1

            frozen = state['frozen']
            m = state['momentum']
            state['step'] += 1

            g_sign = p.grad.sign().float()
            g_sign[frozen] = 0

            m.mul_(0.99).add_(g_sign, alpha=0.01)
            m[frozen] = 0

            mixed = 0.9 * m + 0.1 * g_sign
            update_sign = mixed.sign()

            active = ~frozen
            if active.any():
                p.data[active] -= ternary_lr * update_sign[active]

            threshold = 0.7 * p.data.abs().mean()
            current_ternary = torch.zeros_like(p.data, dtype=torch.int8)
            current_ternary[p.data > threshold] = 1
            current_ternary[p.data < -threshold] = -1

            changed = (current_ternary != state['prev_ternary'])
            n_changes = changed.sum().item()
            if n_changes > 0:
                state['flip_count'] += n_changes
                self.total_flips += n_changes

            state['stability'][~changed] += 1
            state['stability'][changed] = 0
            state['stability'].clamp_(max=32767)

            if freeze_enabled:
                freeze_threshold = group.get('commitment_threshold_steps', 200)
                newly_frozen = (state['stability'] >= freeze_threshold) & ~frozen
                if newly_frozen.any():
                    n_frozen = newly_frozen.sum().item()
                    frozen[newly_frozen] = True
                    self.total_frozen_params += n_frozen

            state['prev_ternary'].copy_(current_ternary)

    def _step_fp16_params(self, group):
        """Update FP16 parameters using SGD with momentum and weight decay."""
        lr = group['lr']
        momentum = group.get('momentum', 0.9)
        weight_decay = group.get('weight_decay', 0.0)

        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(
                    p.data, dtype=torch.float32, device=p.device
                )

            grad = p.grad.data
            if weight_decay > 0:
                grad = grad.add(p.data, alpha=weight_decay)

            buf = state['momentum_buffer']
            buf.mul_(momentum).add_(grad)
            p.data.add_(buf, alpha=-lr)

    def get_stats(self):
        """Return diagnostic statistics.

        Returns:
            dict with keys:
            - 'total_steps': Total optimizer steps
            - 'total_flips': Total ternary weight updates
            - 'flips_per_step': Average flips per step
            - 'frozen_params': Total frozen parameters
            - 'frozen_pct': Percentage of frozen ternary params
            - 'avg_abs_counter': Mean absolute sign counter
            - 'max_counter': Peak sign counter value
        """
        total_params = 0
        total_frozen = 0
        total_abs_counter = 0.0
        max_counter = 0

        for group in self.param_groups:
            is_ternary = group.get('is_ternary', False)

            for p in group['params']:
                if p not in self.state:
                    continue

                state = self.state[p]

                if is_ternary:
                    momentum = state.get('momentum')
                    frozen = state.get('frozen')

                    if momentum is not None:
                        n = momentum.numel()
                        total_params += n

                        abs_momentum = momentum.abs().float()
                        total_abs_counter += abs_momentum.sum().item()
                        max_counter = max(max_counter, abs_momentum.max().item())

                    if frozen is not None:
                        total_frozen += frozen.sum().item()

        avg_abs_counter = (
            total_abs_counter / total_params if total_params > 0 else 0.0
        )
        frozen_pct = 100.0 * total_frozen / total_params if total_params > 0 else 0.0

        return {
            'total_steps': self.total_steps,
            'total_flips': self.total_flips,
            'flips_per_step': self.total_flips / max(1, self.total_steps),
            'frozen_params': total_frozen,
            'frozen_pct': frozen_pct,
            'avg_abs_counter': avg_abs_counter,
            'max_counter': max_counter,
        }

    @torch.no_grad()
    def apply_gradient_mask(self):
        """Zero gradients for committed/frozen ternary weights.

        Call BEFORE optimizer.step() to skip unnecessary gradient computation
        for weights that have already committed to their ternary values.

        Returns:
            float: Fraction of ternary gradients zeroed (0.0 to 1.0)
        """
        total_zeroed = 0
        total_ternary = 0

        for group in self.param_groups:
            if not group.get('is_ternary', False):
                continue

            for p in group['params']:
                if p.grad is None or p not in self.state:
                    continue

                state = self.state[p]
                frozen = state.get('frozen')
                if frozen is None:
                    continue

                p.grad[frozen] = 0

                total_zeroed += frozen.sum().item()
                total_ternary += p.numel()

        return total_zeroed / max(1, total_ternary)

    def get_frozen_mask(self):
        """Return dict mapping param_id -> frozen mask for all ternary params.

        Returns:
            dict: {id(param) -> frozen_mask_tensor} for all ternary parameters
        """
        frozen_masks = {}
        for group in self.param_groups:
            if not group.get('is_ternary', False):
                continue
            for p in group['params']:
                if p in self.state and 'frozen' in self.state[p]:
                    frozen_masks[id(p)] = self.state[p]['frozen']
        return frozen_masks

    def get_stability_stats(self):
        """Return per-parameter stability statistics (consecutive steps at same ternary value).

        Returns:
            dict: {param_id -> stability_counter} for all parameters
        """
        stability_stats = {}
        for group in self.param_groups:
            if not group.get('is_ternary', False):
                continue
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    stability = state.get('stability')
                    if stability is not None:
                        stability_stats[id(p)] = stability

        return stability_stats
