"""Belief Propagation Training for Ternary Networks.

Each weight maintains a 3-element probability vector [P(-1), P(0), P(+1)].
Message passing on the computation graph updates beliefs.

Inspired by LDPC/turbo-code decoding - ternary weights as a
channel coding problem where we decode the optimal weight configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefPropagationTrainer:
    """Trains ternary weights via belief propagation message passing.

    Each weight w_ij has beliefs: [P(w=-1), P(w=0), P(w=+1)]

    Messages from data likelihood update beliefs.
    Weight is set to argmax of belief vector.
    """

    def __init__(
        self, model, damping: float = 0.5, temperature: float = 1.0
    ):
        self.model = model
        self.damping = damping
        self.temperature = temperature
        self.device = next(model.parameters()).device

        # Initialize beliefs (uniform prior)
        self.beliefs = {}
        for name, param in model.named_parameters():
            belief_shape = tuple(param.shape) + (3,)
            self.beliefs[name] = (
                torch.ones(belief_shape, device=self.device) / 3.0
            )

    def compute_likelihood_messages(
        self, loss_fn, x, targets
    ) -> dict:
        """Compute likelihood messages for each weight value.

        For each weight, use gradient as approximate message:
        - Positive gradient -> decrease weight (favor -1)
        - Negative gradient -> increase weight (favor +1)
        - Small gradient -> keep current (favor 0 or current)
        """
        self.model.zero_grad()
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        messages = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()
            msg = torch.zeros(*param.shape, 3, device=self.device)

            # Gradient tells us direction of improvement:
            # Large positive grad -> weight should decrease -> favor -1
            # Large negative grad -> weight should increase -> favor +1
            # Small grad -> weight is OK -> favor current value

            # Message for value -1: benefit from moving toward -1
            msg[..., 0] = torch.clamp(grad, min=0)

            # Message for value 0: benefit from small grad (balanced)
            msg[..., 1] = 1.0 / (1.0 + grad.abs())

            # Message for value +1: benefit from moving toward +1
            msg[..., 2] = torch.clamp(-grad, min=0)

            # Normalize to probability
            msg = F.softmax(msg / self.temperature, dim=-1)
            messages[name] = msg

        return messages

    def update_beliefs(self, messages: dict):
        """Update beliefs using message passing with damping."""
        for name in self.beliefs:
            if name in messages:
                # Damped update: new_belief = damping * old + (1-damping) * message
                self.beliefs[name] = (
                    self.damping * self.beliefs[name]
                    + (1 - self.damping) * messages[name]
                )

                # Renormalize
                belief_sum = self.beliefs[name].sum(dim=-1, keepdim=True).clamp(
                    min=1e-8
                )
                self.beliefs[name] = self.beliefs[name] / belief_sum

    def set_weights_from_beliefs(self):
        """Set ternary weights to argmax of beliefs."""
        values = torch.tensor([-1.0, 0.0, 1.0], device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.beliefs:
                best_arm = self.beliefs[name].argmax(dim=-1)
                new_weight = values[best_arm]
                param.data.copy_(new_weight)

    def train_step(self, loss_fn, x, targets) -> float:
        """One BP training step.

        Returns:
            Current loss (float)
        """
        messages = self.compute_likelihood_messages(loss_fn, x, targets)
        self.update_beliefs(messages)
        self.set_weights_from_beliefs()

        # Compute current loss for logging
        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return loss.item()
