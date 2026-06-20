"""Contrastive Hebbian Learning for Ternary Networks.

Two-phase local learning rule:
1. Positive phase: teacher forcing (correct next token provided)
2. Negative phase: free-running (model generates its own next token)

Weight update: delta_W_ij = sign(correlation_positive - correlation_negative)

For ternary: the sign() operation naturally produces {-1, 0, +1}.
No backward pass, no chain rule, fully local.
"""

import torch
import torch.nn as nn


class ContrastiveHebbianTrainer:
    """Trains ternary network using contrastive Hebbian learning.

    Each weight update is based on the difference in correlations
    between teacher-forced and free-running activations.
    """

    def __init__(self, model, lr: float = 0.01, corruption_rate: float = 0.15):
        self.model = model
        self.lr = lr
        self.corruption_rate = corruption_rate
        self.device = next(model.parameters()).device

        # Running correlation estimates
        self.corr_positive = {}
        self.corr_negative = {}

        for name, param in model.named_parameters():
            self.corr_positive[name] = torch.zeros_like(param.data)
            self.corr_negative[name] = torch.zeros_like(param.data)

    def _collect_activations(self, x: torch.Tensor) -> dict:
        """Collect pre/post activations for each linear layer."""
        activations = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, inp, out):
                if isinstance(inp, tuple):
                    input_act = inp[0].detach()
                else:
                    input_act = inp.detach()

                if torch.is_tensor(out):
                    output_act = out.detach()
                elif isinstance(out, tuple):
                    output_act = out[0].detach()
                else:
                    output_act = out

                activations[name] = {"input": input_act, "output": output_act}

            return hook_fn

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        with torch.no_grad():
            self.model(x)

        for h in hooks:
            h.remove()

        return activations

    def _compute_correlations(self, activations: dict) -> dict:
        """Compute input-output correlations for each layer."""
        correlations = {}

        for name, act in activations.items():
            inp = act["input"]
            out = act["output"]

            # Flatten to 2D: [batch*seq, features]
            inp_flat = inp.reshape(-1, inp.size(-1))
            out_flat = out.reshape(-1, out.size(-1))

            if inp_flat.size(0) == 0 or out_flat.size(0) == 0:
                continue

            # Correlation: normalized outer product average
            # corr[i,j] = mean_over_batch(out_i * inp_j)
            corr = torch.einsum("bi,bj->ij", out_flat, inp_flat) / inp_flat.size(0)
            correlations[name + ".weight"] = corr

        return correlations

    def train_step(
        self, x_positive: torch.Tensor, x_negative: torch.Tensor
    ) -> float:
        """One contrastive Hebbian training step.

        Args:
            x_positive: Teacher-forced input (correct tokens)
            x_negative: Free-running or corrupted input

        Returns:
            Magnitude of weight updates (float)
        """
        # Positive phase: teacher forcing
        pos_activations = self._collect_activations(x_positive)
        pos_correlations = self._compute_correlations(pos_activations)

        # Negative phase: free running
        neg_activations = self._collect_activations(x_negative)
        neg_correlations = self._compute_correlations(neg_activations)

        # Update weights: delta_W = sign(corr_pos - corr_neg)
        total_update_mag = 0.0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in pos_correlations and name in neg_correlations:
                    delta = pos_correlations[name] - neg_correlations[name]

                    # For ternary: use sign
                    update = delta.sign() * self.lr

                    # Match dimensions
                    if update.shape == param.shape:
                        param.data.add_(update)
                        total_update_mag += update.abs().mean().item()

        return total_update_mag

    def generate_negative_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Generate negative sample by corrupting input.

        Simple strategy: randomly replace tokens with random values.
        """
        if isinstance(x, torch.Tensor):
            mask = torch.rand_like(x.float()) < self.corruption_rate
            random_tokens = torch.randint(0, 50257, x.shape, device=x.device)
            negative = torch.where(mask, random_tokens, x)
            return negative
        return x
