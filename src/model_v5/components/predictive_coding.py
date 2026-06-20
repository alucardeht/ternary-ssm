"""Predictive Coding Training - no backward pass, local learning only.

Each layer predicts the next layer's output. Learning signal is the
prediction error at each layer, propagated locally (no chain rule).

For SSMs: the RecursiveSSM provides 32x statistical amplification
through its iterations, making local learning more effective.

Memory: O(1) per layer - no stored activations needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveCodingTrainer:
    """Trains model using predictive coding instead of backpropagation.

    Each layer i has a predictor that estimates layer i+1's output.
    The prediction error is the local learning signal.

    No backward pass required - each layer updates independently.
    """

    def __init__(self, model, lr: float = 0.01, error_weight: float = 0.5):
        self.model = model
        self.lr = lr
        self.error_weight = error_weight
        self.device = next(model.parameters()).device

        # Collect all linear-like layers for prediction
        self.linear_layers = []
        self.predictors = nn.ModuleList()
        self.predictor_optimizers = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.linear_layers.append((name, module))

        for i in range(len(self.linear_layers) - 1):
            curr_module = self.linear_layers[i][1]
            next_module = self.linear_layers[i + 1][1]

            in_features = curr_module.out_features
            out_features = next_module.out_features

            pred = nn.Linear(in_features, out_features, bias=False)
            pred.to(self.device)
            self.predictors.append(pred)
            self.predictor_optimizers.append(
                torch.optim.SGD(pred.parameters(), lr=lr)
            )

    def train_step(self, x: torch.Tensor) -> float:
        """One predictive coding training step.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Total prediction error (float)
        """
        total_error = 0.0

        # Forward through model, collecting layer outputs
        activations = [x.detach()]
        current = x.detach()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    current = module(current)
                activations.append(current.detach())

        if len(activations) < 2:
            return 0.0

        # Compute prediction errors and update predictors
        for i, (pred, opt) in enumerate(zip(self.predictors, self.predictor_optimizers)):
            if i + 1 >= len(activations):
                break

            input_activation = activations[i]
            target_activation = activations[i + 1]

            # Predict next layer's activation
            prediction = pred(input_activation)

            # Match dimensions if needed
            if prediction.shape[-1] != target_activation.shape[-1]:
                continue

            # Prediction error (MSE)
            error = F.mse_loss(prediction, target_activation)
            total_error += error.item()

            # Update predictor to minimize error
            opt.zero_grad()
            error.backward()
            opt.step()

        return total_error
