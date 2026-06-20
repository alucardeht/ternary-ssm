import torch
from collections import deque
from typing import Tuple, Optional, Dict


class CommitmentEarlyStopping:

    def __init__(
        self,
        min_training_fraction: float = 0.5,
        patience_steps: int = 1000,
        commitment_threshold: float = 0.97,
        commitment_rate_threshold: float = 0.0001,
        loss_delta_threshold: float = 0.001,
        flip_rate_threshold: float = 0.001,
    ):
        self.min_training_fraction = min_training_fraction
        self.patience = patience_steps

        self.commitment_threshold = commitment_threshold
        self.commitment_rate_threshold = commitment_rate_threshold
        self.loss_delta_threshold = loss_delta_threshold
        self.flip_rate_threshold = flip_rate_threshold

        self.commitment_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=100)
        self.flip_rate_history = deque(maxlen=1000)
        self.consecutive_satisfied = 0

    def check_stopping(
        self,
        step: int,
        total_steps: int,
        commitment: float,
        flip_rate: float,
        loss: float,
    ) -> Tuple[bool, Optional[str]]:

        if step / total_steps < self.min_training_fraction:
            return False, None

        self.commitment_history.append(commitment)
        self.loss_history.append(loss)
        self.flip_rate_history.append(flip_rate)

        commitment_rate = 0.0
        if len(self.commitment_history) >= 1000:
            commitment_rate = (
                self.commitment_history[-1] - self.commitment_history[0]
            ) / 1000

        loss_delta = 0.0
        if len(self.loss_history) >= 100:
            loss_delta = self.loss_history[0] - self.loss_history[-1]

        conditions = {
            "commitment": commitment > self.commitment_threshold,
            "commitment_rate": commitment_rate < self.commitment_rate_threshold,
            "loss_delta": loss_delta < self.loss_delta_threshold,
            "flip_rate": flip_rate < self.flip_rate_threshold,
        }

        all_satisfied = all(conditions.values())

        if all_satisfied:
            self.consecutive_satisfied += 1
        else:
            self.consecutive_satisfied = 0

        if self.consecutive_satisfied >= self.patience:
            reason = (
                f"Early stop at step {step}: "
                f"C={commitment:.4f}, "
                f"dC/dt={commitment_rate:.6f}, "
                f"ΔL={loss_delta:.6f}, "
                f"flip_rate={flip_rate:.6f}"
            )
            return True, reason

        return False, None

    def get_status(self) -> Dict[str, float]:

        commitment = (
            self.commitment_history[-1] if self.commitment_history else 0.0
        )
        flip_rate = self.flip_rate_history[-1] if self.flip_rate_history else 0.0

        commitment_rate = 0.0
        if len(self.commitment_history) >= 1000:
            commitment_rate = (
                self.commitment_history[-1] - self.commitment_history[0]
            ) / 1000

        loss_delta = 0.0
        if len(self.loss_history) >= 100:
            loss_delta = self.loss_history[0] - self.loss_history[-1]

        return {
            "commitment": commitment,
            "commitment_rate": commitment_rate,
            "loss_delta": loss_delta,
            "flip_rate": flip_rate,
            "consecutive_satisfied": self.consecutive_satisfied,
        }
