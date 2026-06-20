"""Neural Checkpoint Codec (R4-A4-P2).

Tiny neural net learns to GENERATE weight values from position encoding.
200K params (800 KB) replaces entire 7B model checkpoint.
At 99.9% accuracy: ~30 MB corrections needed.
"""
import torch
import torch.nn as nn
import math


class NeuralCheckpointCodec(nn.Module):
    """Small neural network that learns to predict weight values from position."""

    def __init__(self, total_params: int, hidden_dim: int = 256,
                 n_layers: int = 3, encoding_dim: int = 64):
        super().__init__()
        self.total_params = total_params
        self.encoding_dim = encoding_dim

        self.register_buffer('freqs',
            torch.exp(torch.arange(0, encoding_dim, 2).float() *
                     -(math.log(10000.0) / encoding_dim)))

        layers = []
        in_dim = encoding_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else 3
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def encode_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal position encodings."""
        pos = positions.float().unsqueeze(-1)
        angles = pos * self.freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Predict ternary class logits for given positions.

        Returns logits of shape (N, 3) for classes [-1, 0, +1].
        """
        encoding = self.encode_positions(positions)
        return self.mlp(encoding)

    def predict_weights(self, positions: torch.Tensor) -> torch.Tensor:
        """Predict ternary weight values."""
        logits = self.forward(positions)
        classes = logits.argmax(dim=-1)
        return classes.float() - 1

    def train_codec(self, flat_weights: torch.Tensor, epochs: int = 100,
                     lr: float = 1e-3, batch_size: int = 4096) -> dict:
        """Train the codec to memorize weight values.

        Args:
            flat_weights: flattened ternary weights {-1, 0, +1}
            epochs: training epochs
            lr: learning rate
            batch_size: batch size for training

        Returns:
            dict with training stats
        """
        device = flat_weights.device
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        targets = (flat_weights + 1).long()
        n = len(targets)
        positions = torch.arange(n, device=device)

        best_acc = 0.0
        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            total_loss = 0
            total_correct = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                logits = self.forward(positions[idx])
                loss = nn.functional.cross_entropy(logits, targets[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * (end - start)
                total_correct += (logits.argmax(-1) == targets[idx]).sum().item()

            acc = total_correct / n
            best_acc = max(best_acc, acc)

        with torch.no_grad():
            all_logits = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                all_logits.append(self.forward(positions[start:end]))
            all_logits = torch.cat(all_logits)
            predicted = all_logits.argmax(-1)
            errors = (predicted != targets)
            n_errors = errors.sum().item()

        return {
            'best_accuracy': best_acc,
            'final_accuracy': total_correct / n,
            'n_errors': n_errors,
            'error_rate': n_errors / n,
            'codec_params': sum(p.numel() for p in self.parameters()),
            'codec_bytes': sum(p.numel() * 4 for p in self.parameters()),
            'corrections_bytes': n_errors * 5,
        }

    def compress_checkpoint(self, flat_weights: torch.Tensor) -> dict:
        """Compress weights using trained codec."""
        device = flat_weights.device
        n = len(flat_weights)
        targets = (flat_weights + 1).long()
        positions = torch.arange(n, device=device)

        with torch.no_grad():
            predictions = self.predict_weights(positions) + 1
            errors = (predictions.long() != targets)
            error_positions = errors.nonzero(as_tuple=True)[0]
            error_values = flat_weights[error_positions].to(torch.int8)

        return {
            'codec_state': self.state_dict(),
            'error_positions': error_positions.to(torch.int32),
            'error_values': error_values,
            'total_params': n,
        }
