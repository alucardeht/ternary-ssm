"""
Predictive Freezer - Predicts parameter commitment and pre-freezes likely candidates.

References:
- R4-A6-M3: Early freezing prediction based on commitment trajectory
- R4-A6-M8: Historical commitment tracking for trajectory analysis

Principle:
  Some parameters commit (freeze) predictably based on historical patterns.
  If a parameter has consistently increased its commitment score over the past 500 steps,
  it's likely to commit in the next 500 steps.

  Pre-freezing candidates:
  1. Identifies parameters on strong commitment trajectories
  2. Predicts when they'll cross the commitment threshold
  3. Freezes them early to save memory and computation
"""

from typing import Dict, List, Optional, Tuple
import math


class PredictiveFreezer:
    """
    Tracks commitment trajectories and predicts which parameters will commit soon.
    Enables proactive freezing before parameters actually reach the threshold.
    """

    def __init__(
        self,
        prediction_horizon: int = 500,
        prediction_confidence: float = 0.9,
        trajectory_window: int = 100,
    ):
        """
        Args:
            prediction_horizon: Look-ahead steps for prediction
            prediction_confidence: Confidence threshold for freezing (0 to 1)
            trajectory_window: Number of recent steps to use for trajectory fitting
        """
        assert 0 < prediction_confidence <= 1.0, "confidence must be in (0, 1]"

        self.prediction_horizon = prediction_horizon
        self.prediction_confidence = prediction_confidence
        self.trajectory_window = trajectory_window

        # Per-parameter trajectory history: param_id -> list of (step, score)
        self._param_trajectories = {}  # Dict[str, List[Tuple[int, float]]]

        # Prediction cache: param_id -> (prediction_prob, predicted_step)
        self._predictions = {}

        self._step_counter = 0

    def update_trajectory(self, param_name: str, commitment_score: float):
        """
        Record the commitment score of a parameter at the current step.

        Args:
            param_name: Parameter identifier
            commitment_score: Current commitment score (higher = closer to frozen)
        """
        if param_name not in self._param_trajectories:
            self._param_trajectories[param_name] = []

        self._param_trajectories[param_name].append((self._step_counter, commitment_score))

        # Keep only recent window to avoid memory bloat
        if len(self._param_trajectories[param_name]) > self.trajectory_window * 2:
            self._param_trajectories[param_name] = self._param_trajectories[param_name][
                -self.trajectory_window :
            ]

    def predict_commitment(
        self, param_name: str, horizon_steps: Optional[int] = None
    ) -> float:
        """
        Predict probability that a parameter will commit within horizon_steps.

        Uses linear regression on recent trajectory to estimate time-to-commit.

        Args:
            param_name: Parameter identifier
            horizon_steps: Prediction horizon (defaults to self.prediction_horizon)

        Returns:
            Probability in [0, 1] that parameter commits within horizon
        """
        if horizon_steps is None:
            horizon_steps = self.prediction_horizon

        if param_name not in self._param_trajectories:
            return 0.0

        trajectory = self._param_trajectories[param_name]
        if len(trajectory) < 2:
            return 0.0

        # Extract recent history for trajectory fitting
        recent = trajectory[-self.trajectory_window :]
        steps = [t[0] for t in recent]
        scores = [t[1] for t in recent]

        # Simple linear regression to estimate score_velocity (d_score / d_step)
        if len(steps) < 2:
            return 0.0

        step_diffs = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
        score_diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]

        avg_velocity = (
            sum(score_diffs) / sum(step_diffs) if sum(step_diffs) > 0 else 0.0
        )

        # Estimate: at what step will commitment score reach 1.0?
        current_score = scores[-1] if scores else 0.0
        commitment_threshold = 1.0

        if avg_velocity > 1e-6:
            steps_to_commit = (commitment_threshold - current_score) / avg_velocity
        else:
            # Parameter not improving, unlikely to commit
            return 0.0

        # Probability = how confident are we it commits within horizon?
        # If steps_to_commit < horizon_steps, probability is high
        if steps_to_commit < 0:
            # Already past threshold
            return 1.0
        elif steps_to_commit < horizon_steps:
            # Will commit within horizon - probability based on margin
            margin = horizon_steps - steps_to_commit
            probability = 1.0 / (1.0 + math.exp(-margin / 100.0))  # Sigmoid
            return probability
        else:
            # Will commit beyond horizon
            return 0.0

    def get_prefreeze_candidates(
        self, confidence_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Return list of parameters that should be pre-frozen based on predictions.

        A parameter is a pre-freeze candidate if:
        1. Its predicted commitment probability >= confidence_threshold
        2. It's on a strong upward trajectory

        Args:
            confidence_threshold: Prediction confidence threshold
                (defaults to self.prediction_confidence)

        Returns:
            List of parameter names recommended for pre-freezing
        """
        if confidence_threshold is None:
            confidence_threshold = self.prediction_confidence

        candidates = []

        for param_name in self._param_trajectories:
            prob = self.predict_commitment(param_name)

            if prob >= confidence_threshold:
                candidates.append(param_name)

        return candidates

    def get_prediction_accuracy(self) -> Dict:
        """
        Return metrics on prediction accuracy (actual vs predicted).

        Compares past predictions to actual commitment outcomes.

        Returns:
            Dict with prediction accuracy statistics
        """
        if not self._predictions:
            return {
                'predictions_made': 0,
                'predictions_confirmed': 0,
                'predictions_failed': 0,
                'accuracy': 0.0,
                'avg_error_steps': 0.0,
            }

        confirmed = 0
        failed = 0
        total_error = 0.0

        for param_name, (pred_prob, pred_step) in self._predictions.items():
            if param_name not in self._param_trajectories:
                continue

            trajectory = self._param_trajectories[param_name]
            if not trajectory:
                continue

            final_score = trajectory[-1][1]

            # Check if prediction was correct
            if final_score >= 1.0:
                confirmed += 1
                # Error: how many steps off was the prediction?
                actual_step = trajectory[-1][0]
                error = abs(actual_step - pred_step)
                total_error += error
            else:
                failed += 1

        total_predictions = confirmed + failed
        accuracy = confirmed / total_predictions if total_predictions > 0 else 0.0
        avg_error = total_error / confirmed if confirmed > 0 else 0.0

        return {
            'predictions_made': total_predictions,
            'predictions_confirmed': confirmed,
            'predictions_failed': failed,
            'accuracy': accuracy,
            'avg_error_steps': avg_error,
            'confidence_threshold': self.prediction_confidence,
        }

    def advance_step(self):
        """Increment the internal step counter."""
        self._step_counter += 1

    def get_trajectory_info(self, param_name: str) -> Dict:
        """
        Return detailed trajectory information for a parameter.

        Args:
            param_name: Parameter identifier

        Returns:
            Dict with trajectory stats: length, velocity, current_score, etc.
        """
        if param_name not in self._param_trajectories:
            return {
                'param_name': param_name,
                'trajectory_length': 0,
                'avg_velocity': 0.0,
                'current_score': 0.0,
            }

        trajectory = self._param_trajectories[param_name]
        steps = [t[0] for t in trajectory]
        scores = [t[1] for t in trajectory]

        # Compute velocity
        if len(steps) < 2:
            avg_velocity = 0.0
        else:
            step_diffs = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
            score_diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
            total_diffs = sum(step_diffs)
            avg_velocity = sum(score_diffs) / total_diffs if total_diffs > 0 else 0.0

        return {
            'param_name': param_name,
            'trajectory_length': len(trajectory),
            'current_score': scores[-1] if scores else 0.0,
            'first_score': scores[0] if scores else 0.0,
            'avg_velocity': avg_velocity,
            'steps_covered': steps[-1] - steps[0] if len(steps) > 1 else 0,
            'recent_trend': (
                'improving' if (scores[-1] > scores[0]) else 'stagnant'
            ),
        }
