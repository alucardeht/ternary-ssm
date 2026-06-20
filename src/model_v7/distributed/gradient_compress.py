"""Gradient compression with error feedback for ternary networks.

Ternary gradients are naturally sparse (many near-zero due to quantization).
We exploit this by transmitting only top-k% largest gradient magnitudes.

Error feedback: residuals from sparsification are accumulated and added
to the next communication round, ensuring no information loss over time
(equivalent to full-precision gradients with arbitrary latency).
"""

import torch


class TernaryGradientCompressor:
    """Sparse gradient compression optimized for ternary networks.

    Compresses gradients using top-k sparsification by magnitude.
    Maintains error feedback buffers to accumulate residuals,
    ensuring convergence equivalent to full-precision gradient descent.

    Args:
        compression_ratio: Fraction of gradients to communicate (default: 0.01 = 1%)
        use_error_feedback: Enable error feedback accumulation (default: True)
    """

    def __init__(self, compression_ratio=0.01, use_error_feedback=True):
        self.compression_ratio = compression_ratio
        self.use_error_feedback = use_error_feedback
        self.error_feedback = {}
        self.compression_stats = {
            'total_elements': 0,
            'total_communicated': 0,
            'rounds': 0,
        }

    def compress(self, param_name, gradient):
        """Compress gradient using top-k sparsification with error feedback.

        Algorithm:
        1. Add error feedback residual from previous round
        2. Compute top-k% by absolute magnitude
        3. Store remainder as error feedback for next round
        4. Return sparse representation: indices and values

        Args:
            param_name: Identifier for this parameter (string)
            gradient: Full gradient tensor (any shape)

        Returns:
            sparse_grad: Dict with keys:
                        - 'indices': List of indices of communicated gradients
                        - 'values': Communicated gradient values
                        - 'shape': Original shape for reconstruction
                        - 'param_name': Parameter identifier
        """
        flat_grad = gradient.reshape(-1).clone()

        if self.use_error_feedback and param_name in self.error_feedback:
            residual = self.error_feedback[param_name]
            flat_grad = flat_grad + residual

        k = max(1, int(flat_grad.numel() * self.compression_ratio))

        magnitudes = flat_grad.abs()
        threshold = torch.kthvalue(magnitudes, magnitudes.numel() - k + 1).values

        mask = magnitudes >= threshold
        if mask.sum().item() != k:
            tied_indices = torch.where(magnitudes == threshold)[0]
            num_tied = tied_indices.numel()
            need = k - (mask.sum().item() - num_tied)
            if need > 0 and num_tied > 0:
                perm = torch.randperm(num_tied)
                mask[tied_indices[perm[:need]]] = True
            mask[tied_indices[perm[need:]]] = False if need > 0 else mask[tied_indices]

        communicated = flat_grad * mask
        residual = flat_grad * (~mask)

        if self.use_error_feedback:
            self.error_feedback[param_name] = residual.clone()

        indices = torch.where(mask)[0].tolist()
        values = communicated[mask].detach().cpu()

        self.compression_stats['total_elements'] += flat_grad.numel()
        self.compression_stats['total_communicated'] += len(indices)

        return {
            'indices': indices,
            'values': values,
            'shape': gradient.shape,
            'param_name': param_name,
        }

    def decompress(self, sparse_grad, device='cpu'):
        """Reconstruct full gradient from sparse representation.

        Args:
            sparse_grad: Dict returned by compress()
            device: Device to place reconstructed gradient on

        Returns:
            full_grad: Tensor with same shape as original, zero-filled where not communicated
        """
        shape = sparse_grad['shape']
        indices = sparse_grad['indices']
        values = sparse_grad['values'].to(device)

        total_elements = 1
        for dim in shape:
            total_elements *= dim

        full_grad = torch.zeros(
            total_elements,
            dtype=values.dtype,
            device=device,
        )

        for i, idx in enumerate(indices):
            full_grad[idx] = values[i]

        return full_grad.reshape(shape)

    def reset_error_feedback(self, param_name=None):
        """Reset error feedback buffers.

        Args:
            param_name: If provided, reset only this parameter's buffer.
                       If None, reset all buffers.
        """
        if param_name is not None:
            if param_name in self.error_feedback:
                del self.error_feedback[param_name]
        else:
            self.error_feedback.clear()

    def get_compression_stats(self):
        """Return compression statistics.

        Returns:
            dict: Statistics including actual compression ratio, communication efficiency
        """
        total = self.compression_stats['total_elements']
        communicated = self.compression_stats['total_communicated']

        if total == 0:
            return {
                'compression_ratio': 1.0,
                'communication_ratio': 1.0,
                'total_elements': 0,
                'total_communicated': 0,
                'rounds': 0,
            }

        actual_ratio = communicated / total if total > 0 else 1.0
        comm_ratio = self.compression_ratio

        return {
            'compression_ratio': self.compression_ratio,
            'actual_ratio': actual_ratio,
            'communication_ratio': comm_ratio,
            'total_elements': total,
            'total_communicated': communicated,
            'rounds': self.compression_stats['rounds'],
            'efficiency': actual_ratio / comm_ratio if comm_ratio > 0 else 1.0,
        }

    def increment_round(self):
        """Increment communication round counter."""
        self.compression_stats['rounds'] += 1

    def get_error_feedback_magnitude(self):
        """Return current magnitude of accumulated error feedback.

        Returns:
            dict: {param_name -> L2 norm of error feedback tensor}
        """
        magnitudes = {}
        for param_name, residual in self.error_feedback.items():
            if residual is not None:
                magnitudes[param_name] = residual.norm().item()

        return magnitudes
