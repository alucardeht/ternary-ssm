"""
Commitment Curriculum - Training schedule that encourages parameter commitment progression.

References:
- R4-A6-P7: Commitment curriculum follows a predefined schedule that gradually increases
  the fraction of parameters expected to be committed (frozen) as training progresses.

Principle:
  Early training: let all parameters explore (low target commitment ratio).
  Mid training: encourage some params to commit (medium target commitment ratio).
  Late training: encourage aggressive freezing (high target commitment ratio).

This schedule is curriculum-based - it guides the model toward a target distribution
of committed parameters, accelerating convergence and memory savings.
"""

from typing import Optional
import math


class CommitmentCurriculum:
    """
    Curriculum scheduler that sets target commitment ratios and bonuses
    as a function of training progress.

    Motivations:
    1. Early freezing saves memory sooner
    2. Commitment is irreversible - choosing the right time matters
    3. A curriculum smooths the transition from exploration to exploitation
    """

    def __init__(
        self,
        total_steps: int,
        curriculum_target_ratio: float = 0.7,
        curriculum_warmup_fraction: float = 0.2,
        commitment_bonus_max: float = 0.2,
    ):
        """
        Args:
            total_steps: Total number of training steps
            curriculum_target_ratio: Final target commitment ratio (0 to 1)
                At step=0: target=0.0, at step=total_steps: target=curriculum_target_ratio
            curriculum_warmup_fraction: Fraction of training before commitment starts
                Warmup region: target stays at 0.0 for this fraction of steps
            commitment_bonus_max: Maximum bonus added to commitment scores
                Encourages parameters near the threshold to commit
        """
        assert 0 <= curriculum_target_ratio <= 1.0, "target_ratio must be in [0, 1]"
        assert 0 <= curriculum_warmup_fraction < 1.0, "warmup_fraction must be in [0, 1)"
        assert commitment_bonus_max >= 0, "bonus_max must be >= 0"

        self.total_steps = total_steps
        self.curriculum_target_ratio = curriculum_target_ratio
        self.curriculum_warmup_fraction = curriculum_warmup_fraction
        self.commitment_bonus_max = commitment_bonus_max

        self.warmup_steps = int(total_steps * curriculum_warmup_fraction)
        self.active_steps = total_steps - self.warmup_steps

    def get_target_commitment_ratio(self, current_step: int) -> float:
        """
        Return the target fraction of parameters that should be committed at this step.

        Args:
            current_step: Current training step (0-indexed)

        Returns:
            Target commitment ratio in [0, curriculum_target_ratio]
        """
        if current_step < self.warmup_steps:
            return 0.0

        progress = (current_step - self.warmup_steps) / max(self.active_steps, 1)
        progress = min(progress, 1.0)

        return self.curriculum_target_ratio * progress

    def get_commitment_bonus(self, current_step: int) -> float:
        """
        Return the bonus to add to commitment scores at this step.

        This bonus encourages parameters near the commitment threshold to cross it.
        Increases as training progresses and the curriculum target increases.

        Args:
            current_step: Current training step

        Returns:
            Bonus value to add to commitment scores (0 during warmup, increases after)
        """
        target_ratio = self.get_target_commitment_ratio(current_step)

        if target_ratio < 1e-6:
            return 0.0

        # Bonus scales with progress: 0 during warmup, ramps up as target_ratio increases
        progress_toward_target = min(target_ratio / self.curriculum_target_ratio, 1.0)

        return self.commitment_bonus_max * progress_toward_target

    def should_encourage_commitment(
        self, current_ratio: float, target_ratio: float
    ) -> bool:
        """
        Check if commitment should be actively encouraged at this stage.

        Args:
            current_ratio: Current fraction of committed parameters
            target_ratio: Target fraction of committed parameters

        Returns:
            True if current_ratio < target_ratio (keep freezing)
            False if current_ratio >= target_ratio (hold pattern)
        """
        return current_ratio < target_ratio

    def get_schedule_info(self, current_step: int) -> dict:
        """
        Return full schedule information for the current step.

        Args:
            current_step: Current training step

        Returns:
            Dict with target_ratio, bonus, warmup_active, and progress metrics
        """
        in_warmup = current_step < self.warmup_steps
        target_ratio = self.get_target_commitment_ratio(current_step)
        bonus = self.get_commitment_bonus(current_step)

        if in_warmup:
            schedule_progress = 0.0
        else:
            schedule_progress = (current_step - self.warmup_steps) / max(
                self.active_steps, 1
            )
            schedule_progress = min(schedule_progress, 1.0)

        return {
            'step': current_step,
            'target_commitment_ratio': target_ratio,
            'commitment_bonus': bonus,
            'in_warmup': in_warmup,
            'schedule_progress': schedule_progress,
            'warmup_steps': self.warmup_steps,
            'active_steps': self.active_steps,
            'total_steps': self.total_steps,
        }
