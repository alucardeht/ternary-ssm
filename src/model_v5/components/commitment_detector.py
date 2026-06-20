"""
Commitment Detection Infrastructure (R3-A6-M1, R3-A6-M4)

Detects when ternary weights have "committed" to their quantized values - i.e., when
shadow weights are far from quantization thresholds and unlikely to flip.

Uses Adam statistics (first/second moment) to model weight updates as biased Brownian
motion. Estimates probability of flipping to different quantization level within horizon K.

Provides per-weight flip probability and block-aligned commitment classification.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class CommitmentDetector:
    """
    Detects weight commitment based on Adam statistics and quantization boundaries.

    Key insight: A shadow weight w follows drift + diffusion near quantization threshold.
    - Drift: K * lr * |m| / (sqrt(v) + eps)
    - Diffusion: sqrt(K) * lr * sqrt(C * (1 - m^2/v))

    P(flip) = P(cross boundary within K steps) ~= error function of distance/volatility.
    If P(flip) < epsilon, weight is committed (safe to freeze/compress/replace).
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        horizon_k: int = 50,
        check_interval: int = 100,
        block_size: int = 128,
    ):
        """
        Args:
            epsilon: P(flip) threshold below which weight is committed (default 0.01 = 1%)
            horizon_k: Steps ahead to predict flip probability
            check_interval: Steps between commitment evaluations
            block_size: Block size for group commitment (must match optimizer: 128)
        """
        self.epsilon = epsilon
        self.horizon_k = horizon_k
        self.check_interval = check_interval
        self.block_size = block_size

        self.flip_history = {}
        self.step_count = 0

    @staticmethod
    def compute_threshold_distance(
        weight: torch.Tensor,
        group_size: int = 128,
        threshold_coeff: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute distance from each weight to nearest quantization boundary.

        For ternary {-1, 0, +1}, boundaries are at ±threshold_coeff.
        Distance = min(|w| - threshold, threshold - |w|) for each weight.

        Args:
            weight: Shadow weights [N]
            group_size: Group size for per-group threshold computation (unused for now, symmetric)
            threshold_coeff: Threshold offset from quantization levels (0.5 = default)

        Returns:
            Distance tensor [N], shape matches input. Larger = further from boundary.
        """
        abs_w = torch.abs(weight)
        threshold = threshold_coeff

        distance_to_boundary = threshold - abs_w
        return distance_to_boundary.clamp(min=0.0)

    @staticmethod
    def compute_flip_probability(
        shadow_weights: torch.Tensor,
        adam_m: torch.Tensor,
        adam_v: torch.Tensor,
        lr: float = 1e-3,
        K: int = 50,
        beta1: float = 0.9,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute probability of weight flipping within K steps using Adam statistics.

        Models weight update as biased Brownian motion:
        - Drift: mu = K * lr * |m_hat| / (sqrt(v_hat) + eps)
        - Diffusion: sigma = sqrt(K) * lr * sqrt(C * (1 - (m_hat/v_hat)^2))

        P(flip) ~= Phi((-d + mu) / sigma) + Phi((-d - mu) / sigma)  [both boundaries]

        Args:
            shadow_weights: Current shadow weights [N]
            adam_m: First moment (mean) from Adam [N]
            adam_v: Second moment (variance) from Adam [N]
            lr: Learning rate
            K: Horizon in steps
            beta1: Adam beta1 parameter (for bias correction)
            eps: Numerical stability epsilon

        Returns:
            P_flip tensor [N], probability each weight flips within K steps.
        """
        m_hat = adam_m / (1.0 - beta1 ** K + eps)
        v_hat = adam_v / (1.0 - 0.999 ** K + eps)

        drift = K * lr * torch.abs(m_hat) / (torch.sqrt(v_hat) + eps)

        variance_factor = 1.0 - torch.clamp((m_hat ** 2) / (v_hat + eps), max=0.999)
        diffusion = math.sqrt(K) * lr * torch.sqrt(torch.clamp(variance_factor, min=0.0))

        distance = CommitmentDetector.compute_threshold_distance(shadow_weights)

        if diffusion.max() < eps:
            p_flip = torch.zeros_like(drift)
        else:
            drift_normalized = (drift - distance) / (diffusion + eps)
            p_flip_pos = 0.5 * (1.0 + torch.erf(drift_normalized / math.sqrt(2.0)))

            drift_normalized_neg = (-drift - distance) / (diffusion + eps)
            p_flip_neg = 0.5 * (1.0 + torch.erf(drift_normalized_neg / math.sqrt(2.0)))

            p_flip = p_flip_pos + p_flip_neg
            p_flip = torch.clamp(p_flip, min=0.0, max=1.0)

        return p_flip

    @staticmethod
    def compute_block_commitment(
        p_flip_flat: torch.Tensor,
        block_size: int = 128,
        epsilon: float = 0.01,
    ) -> torch.Tensor:
        """
        Classify blocks of weights based on flip probabilities.

        Per-block commitment = (max P_flip in block < epsilon).
        Conservative: if ANY weight in block might flip, block is ACTIVE.

        Args:
            p_flip_flat: Flat flip probability tensor [N]
            block_size: Block size (must match optimizer, typically 128)
            epsilon: Threshold for commitment

        Returns:
            Boolean tensor [num_blocks], True = committed (safe to freeze/compress).
        """
        N = p_flip_flat.shape[0]
        num_blocks = (N + block_size - 1) // block_size

        block_commitment = torch.ones(num_blocks, dtype=torch.bool, device=p_flip_flat.device)

        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, N)
            block_max_flip = p_flip_flat[start:end].max()
            block_commitment[i] = (block_max_flip < epsilon)

        return block_commitment

    def update_flip_history(
        self,
        param_name: str,
        old_ternary: torch.Tensor,
        new_ternary: torch.Tensor,
    ) -> None:
        """
        Track weight flips over time for metabolic rate estimation.

        Args:
            param_name: Name of parameter (e.g., "layer0.weight")
            old_ternary: Previous quantized weights {-1, 0, +1}
            new_ternary: Current quantized weights {-1, 0, +1}
        """
        if param_name not in self.flip_history:
            self.flip_history[param_name] = {
                'flips': [],
                'steps': [],
            }

        flips = (old_ternary != new_ternary).float()
        self.flip_history[param_name]['flips'].append(flips)
        self.flip_history[param_name]['steps'].append(self.step_count)

        if len(self.flip_history[param_name]['flips']) > 1000:
            self.flip_history[param_name]['flips'].pop(0)
            self.flip_history[param_name]['steps'].pop(0)

    def get_metabolic_rate(
        self,
        param_name: str,
        window: int = 100,
    ) -> float:
        """
        Estimate metabolic rate (flip rate) over recent window.

        Low metabolic rate = weight is committed (not flipping).
        High rate = weight is active (still learning).

        Args:
            param_name: Name of parameter
            window: Number of recent steps to consider

        Returns:
            Flips per step over window (0.0 = no flips, 1.0 = every step).
        """
        if param_name not in self.flip_history or len(self.flip_history[param_name]['flips']) == 0:
            return 0.0

        flips = self.flip_history[param_name]['flips'][-window:]
        if len(flips) == 0:
            return 0.0

        stack = torch.stack(flips)
        return stack.mean().item()

    def classify_blocks(
        self,
        p_flip_flat: torch.Tensor,
        metabolic_rates: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Classify blocks into three categories: COMMITTED, ACTIVE, TRANSITIONING.

        Args:
            p_flip_flat: Flip probability tensor [N]
            metabolic_rates: Optional dict of metabolic rates per layer (for validation)

        Returns:
            Dict with keys:
                - 'committed': BoolTensor [num_blocks], safe to freeze
                - 'active': BoolTensor [num_blocks], still learning
                - 'transitioning': BoolTensor [num_blocks], uncertain
        """
        block_commitment = self.compute_block_commitment(
            p_flip_flat,
            block_size=self.block_size,
            epsilon=self.epsilon,
        )

        p_flip_max_per_block = []
        N = p_flip_flat.shape[0]
        num_blocks = (N + self.block_size - 1) // self.block_size

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, N)
            p_flip_max_per_block.append(p_flip_flat[start:end].max().item())

        p_flip_max_per_block = torch.tensor(
            p_flip_max_per_block,
            dtype=torch.float32,
            device=p_flip_flat.device,
        )

        transitioning = (p_flip_max_per_block >= self.epsilon * 0.1) & \
                        (p_flip_max_per_block < self.epsilon * 10.0)
        active = p_flip_max_per_block >= (self.epsilon * 10.0)

        return {
            'committed': block_commitment,
            'active': active,
            'transitioning': transitioning,
            'p_flip_max': p_flip_max_per_block,
        }

    def get_frozen_mask(self) -> Optional[torch.Tensor]:
        """
        Get the latest computed frozen (committed) mask.

        Returns:
            BoolTensor [num_blocks] or None if not yet computed.
        """
        if not hasattr(self, '_latest_frozen_mask'):
            return None
        return self._latest_frozen_mask

    def evaluate(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr: float,
    ) -> Dict:
        """
        Evaluate commitment status across all ternary layers.

        Args:
            model: The neural network
            optimizer: The optimizer (to access Adam states m, v)
            lr: Current learning rate

        Returns:
            Dict with per-layer commitment info and global frozen mask.
        """
        results = {
            'layers': [],
            'global_commitment_rate': 0.0,
            'frozen_mask': None,
        }

        all_p_flip = []

        for name, param in model.named_parameters():
            if 'ternary' not in name or param.grad is None:
                continue

            if name not in optimizer.state or len(optimizer.state[name]) == 0:
                continue

            opt_state = optimizer.state[name]

            if 'exp_avg' not in opt_state or 'exp_avg_sq' not in opt_state:
                continue

            adam_m = opt_state['exp_avg']
            adam_v = opt_state['exp_avg_sq']

            p_flip = self.compute_flip_probability(
                shadow_weights=param,
                adam_m=adam_m,
                adam_v=adam_v,
                lr=lr,
                K=self.horizon_k,
            )

            all_p_flip.append(p_flip.flatten())

            committed_count = (p_flip < self.epsilon).sum().item()
            total_count = p_flip.numel()
            committed_frac = committed_count / (total_count + 1e-8)

            results['layers'].append({
                'name': name,
                'committed_frac': committed_frac,
                'p_flip_mean': p_flip.mean().item(),
                'p_flip_max': p_flip.max().item(),
            })

        if all_p_flip:
            all_p_flip_concat = torch.cat(all_p_flip)
            global_commitment = (all_p_flip_concat < self.epsilon).float().mean().item()
            results['global_commitment_rate'] = global_commitment

            frozen_mask = self.compute_block_commitment(
                all_p_flip_concat,
                block_size=self.block_size,
                epsilon=self.epsilon,
            )
            results['frozen_mask'] = frozen_mask
            self._latest_frozen_mask = frozen_mask

        self.step_count += 1
        return results
