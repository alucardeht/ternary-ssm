import torch
import math
import logging

logger = logging.getLogger(__name__)


class CommitRestartScheduler:
    """Reinitialize uncommitted weights to escape local optima.

    Triggered by: (1) regular interval, (2) commitment stalled.
    Preserves committed weights (temperature floor), rehats uncommitted.
    """

    def __init__(self, restart_interval=2000, stuck_threshold=500):
        self.restart_interval = restart_interval
        self.stuck_threshold = stuck_threshold

        self.last_restart_step = 0
        self.last_commitment_change = 0
        self.commitment_count = 0

    def should_restart(self, step, current_commitment_count):
        """Check if restart needed."""
        if step - self.last_restart_step >= self.restart_interval:
            return True

        if current_commitment_count == self.commitment_count:
            if step - self.last_commitment_change >= self.stuck_threshold:
                logger.warning(f"[Restart] Commitment stuck at {self.commitment_count} "
                             f"for {step - self.last_commitment_change} steps")
                return True
        else:
            self.last_commitment_change = step
            self.commitment_count = current_commitment_count

        return False

    def restart_uncommitted(self, optimizer, uncommitted_params, step):
        """Reinitialize uncommitted weights and reset optimizer state."""
        num_restarted = 0

        for p in uncommitted_params:
            if p.dim() < 2:
                fan_in = p.shape[0] if p.dim() > 0 else 1
            else:
                fan_in = p.shape[1]

            std = 1.0 / math.sqrt(max(fan_in, 1))

            with torch.no_grad():
                p.data.normal_(0, std)

            if hasattr(optimizer, 'state') and p in optimizer.state:
                del optimizer.state[p]

            num_restarted += 1

        logger.info(f"[Restart] Step {step}: restarted {num_restarted} weights")
        self.last_restart_step = step

        return num_restarted

    def get_stats(self):
        return {
            'commitment_count': self.commitment_count,
            'last_restart': self.last_restart_step
        }
