import torch
from torch.optim import Optimizer
import math


class BBOOptimizer(Optimizer):
    """Ballot-Box Optimizer for ternary weight training without shadow weights.

    Replaces FP32 shadow weights + Adam states (12 bytes/param) with INT8 ballot
    counters (1 byte/param) for ternary weight layers. Each gradient step casts
    a "vote" (+1 or -1) based on gradient sign. When votes exceed threshold K,
    the ternary weight transitions to the next state.

    Voting mechanism:
    - Gradient sign determines vote direction: +1 (positive) or -1 (negative)
    - Votes accumulate in INT8 ballot counter [-128, +127]
    - When |votes| >= K, state transitions and ballot resets
    - Exponential forgetting (periodic halving) prevents stale evidence

    State transitions (with two-step mode):
    - State -1: votes > +K -> flip to 0
    - State  0: votes > +K -> flip to +1
    - State  0: votes < -K -> flip to -1
    - State +1: votes < -K -> flip to 0
    - Two flips required for -1 <-> +1 ensures inertia

    Args:
        params: Iterable of parameters (ternary weights, shape = {-1, 0, +1})
        threshold_K: INT8 votes needed to trigger state transition (default: 16)
        decay_steps: Halve all counters every N steps (0 = disable, default: 100)
        magnitude_threshold: Minimum gradient magnitude to count as vote (default: 0.0)
        use_two_step: Require intermediate state for -1 <-> +1 transitions (default: True)
    """

    def __init__(
        self,
        params,
        threshold_K: int = 16,
        decay_steps: int = 100,
        magnitude_threshold: float = 0.0,
        use_two_step: bool = True,
    ):
        if threshold_K <= 0:
            raise ValueError(f"Invalid threshold_K: {threshold_K}")
        if decay_steps < 0:
            raise ValueError(f"Invalid decay_steps: {decay_steps}")
        if magnitude_threshold < 0.0:
            raise ValueError(f"Invalid magnitude_threshold: {magnitude_threshold}")

        defaults = dict(
            threshold_K=threshold_K,
            decay_steps=decay_steps,
            magnitude_threshold=magnitude_threshold,
            use_two_step=use_two_step,
        )
        super().__init__(params, defaults)
        self._total_steps = 0
        self._total_flips = 0

    def initialize_from_shadow(self, ternary_weights):
        """Initialize ternary states from current weight values."""
        initialized = 0
        for param in ternary_weights:
            state = self.state[param]
            if len(state) == 0:
                state['ballot'] = torch.zeros_like(
                    param.data, dtype=torch.int8, device=param.device
                )
                state['ternary'] = torch.zeros_like(
                    param.data, dtype=torch.int8, device=param.device
                )
                state['step'] = 0

            quantized = torch.sign(param.data).to(torch.int8)
            state['ternary'].copy_(quantized)
            param.data.copy_(quantized.float())
            initialized += 1

        print(
            f"[BBO-INIT] Initialized {initialized} ternary param tensors from shadow weights",
            flush=True,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimizer step.

        For each ternary parameter:
        1. Cast gradient sign as vote (+1 or -1)
        2. Accumulate vote in INT8 ballot (clamped to [-128, +127])
        3. Check thresholds: if |ballot| >= K, transition ternary state and reset
        4. Write ternary state back to param.data
        5. Periodically halve ballots for exponential forgetting

        Args:
            closure: Optional closure for loss computation (unused, for compatibility)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            threshold_K = group['threshold_K']
            decay_steps = group['decay_steps']
            magnitude_threshold = group['magnitude_threshold']
            use_two_step = group['use_two_step']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state['ballot'] = torch.zeros_like(param.data, dtype=torch.int8, device=param.device)
                    state['ternary'] = torch.sign(param.data).to(torch.int8)
                    state['step'] = 0

                state['step'] += 1

                ballot = state['ballot']
                ternary = state['ternary']

                grad_sign = torch.sign(grad).to(torch.int8)

                if magnitude_threshold > 0.0:
                    mask = grad.abs() > magnitude_threshold
                    grad_sign = grad_sign * mask.to(torch.int8)

                ballot.add_(grad_sign)
                ballot.clamp_(-128, 127)

                pos_fire = ballot >= threshold_K
                neg_fire = ballot <= -threshold_K

                new_ternary = ternary.clone()

                if pos_fire.any():
                    if use_two_step:
                        new_ternary = torch.where(
                            pos_fire & (ternary == -1),
                            torch.zeros_like(ternary),
                            new_ternary,
                        )
                        new_ternary = torch.where(
                            pos_fire & (ternary == 0),
                            torch.ones_like(ternary),
                            new_ternary,
                        )
                    else:
                        new_ternary = torch.where(
                            pos_fire,
                            (ternary + 1).clamp(-1, 1),
                            new_ternary,
                        )
                    ballot = torch.where(pos_fire, torch.zeros_like(ballot), ballot)
                    self._total_flips += int((new_ternary != ternary).sum().item())

                if neg_fire.any():
                    if use_two_step:
                        new_ternary = torch.where(
                            neg_fire & (ternary == 1),
                            torch.zeros_like(ternary),
                            new_ternary,
                        )
                        new_ternary = torch.where(
                            neg_fire & (ternary == 0),
                            torch.full_like(ternary, -1),
                            new_ternary,
                        )
                    else:
                        new_ternary = torch.where(
                            neg_fire,
                            (ternary - 1).clamp(-1, 1),
                            new_ternary,
                        )
                    ballot = torch.where(neg_fire, torch.zeros_like(ballot), ballot)
                    self._total_flips += int((new_ternary != ternary).sum().item())

                state['ternary'].copy_(new_ternary)
                state['ballot'].copy_(ballot)
                param.data.copy_(new_ternary.float())

                if self._total_steps % 100 == 0:
                    print(
                        f"[BBO-HOOK] total_steps={self._total_steps} "
                        f"param_shape={param.shape} "
                        f"pos_fire={pos_fire.sum().item()} neg_fire={neg_fire.sum().item()} "
                        f"total_flips={self._total_flips}",
                        flush=True,
                    )

                if decay_steps > 0 and state['step'] % decay_steps == 0:
                    # Convert to float, divide by 2, convert back to int8
                    ballot_float = ballot.float()
                    ballot_float.div_(2)
                    ballot.copy_(ballot_float.to(torch.int8))

        self._total_steps += 1
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none=False):
        """Set gradients to zero (required for training loop compatibility)."""
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def get_stats(self):
        """Return optimizer statistics for monitoring.

        Returns:
            dict with keys:
            - 'total_steps': Total optimizer steps
            - 'total_flips': Total ternary state transitions
            - 'flips_per_step': Average flips per step
            - 'avg_abs_ballot': Mean absolute ballot value across all params
            - 'max_ballot': Peak absolute ballot value
            - 'near_threshold_pct': % of ballots within 1-2 of threshold
            - 'state_distribution': {-1: %, 0: %, +1: %} distribution
        """
        total_params = 0
        total_abs_ballot = 0.0
        max_abs_ballot = 0.0
        near_threshold = 0
        state_counts = {-1: 0, 0: 0, 1: 0}

        for group in self.param_groups:
            threshold_K = group['threshold_K']
            for param in group['params']:
                if param not in self.state:
                    continue

                state = self.state[param]
                ballot = state.get('ballot')
                ternary = state.get('ternary')

                if ballot is None or ternary is None:
                    continue

                total_params += ternary.numel()
                total_abs_ballot += ballot.abs().sum().item()
                max_abs_ballot = max(max_abs_ballot, ballot.abs().max().item())

                near_threshold += ((ballot.abs() >= threshold_K - 2) & (ballot.abs() < threshold_K)).sum().item()

                for state_val in [-1, 0, 1]:
                    state_counts[state_val] += (ternary == state_val).sum().item()

        avg_abs_ballot = total_abs_ballot / max(total_params, 1)
        near_pct = 100.0 * near_threshold / max(total_params, 1)

        state_dist = {}
        for state_val in [-1, 0, 1]:
            state_dist[state_val] = 100.0 * state_counts[state_val] / max(total_params, 1)

        flips_per_step = self._total_flips / max(self._total_steps, 1)

        return {
            'total_steps': self._total_steps,
            'total_flips': self._total_flips,
            'flips_per_step': flips_per_step,
            'avg_abs_ballot': avg_abs_ballot,
            'max_ballot': max_abs_ballot,
            'near_threshold_pct': near_pct,
            'state_distribution': state_dist,
        }
