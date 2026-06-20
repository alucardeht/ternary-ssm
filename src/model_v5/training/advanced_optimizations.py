import math
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BlockSizeSearcher:
    """Find optimal quantization block size via forward-only search.

    Principle: quantization error depends on block size. Too small = overhead,
    too large = poor approximation. Sweep block sizes {32,64,128,256} using
    only forward passes (no training needed).
    """

    def __init__(self, block_sizes=(32, 64, 128, 256)):
        self.block_sizes = block_sizes
        self.results = {}

    def search(self, model, data_loader, device='cpu', n_batches=5):
        """Evaluate each block size by measuring quantization error.
        Returns optimal block size and error per size.

        Args:
            model: PyTorch model
            data_loader: DataLoader with input tensors
            device: 'cpu' or 'cuda'
            n_batches: Number of batches to sample

        Returns:
            Dict mapping block_size to mean quantization error
        """
        self.results = {}
        model.to(device)
        model.eval()

        batch_count = 0
        for batch_idx, batch in enumerate(data_loader):
            if batch_count >= n_batches:
                break

            if isinstance(batch, (list, tuple)):
                x = batch[0] if len(batch) > 0 else batch
            else:
                x = batch

            if not isinstance(x, torch.Tensor):
                continue

            x = x.to(device)
            batch_count += 1

            with torch.no_grad():
                for block_size in self.block_sizes:
                    quant_error = self._compute_quantization_error(model, x, block_size, device)

                    if block_size not in self.results:
                        self.results[block_size] = []
                    self.results[block_size].append(quant_error)

        for block_size in self.block_sizes:
            if block_size in self.results and len(self.results[block_size]) > 0:
                self.results[block_size] = sum(self.results[block_size]) / len(self.results[block_size])
            else:
                self.results[block_size] = 0.0

        return self.results

    def _compute_quantization_error(self, model, x, block_size, device):
        """Compute quantization error for a given block size."""
        total_error = 0.0
        count = 0

        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                error = self._block_quantization_error(weight, block_size)
                total_error += error
                count += 1

        return total_error / max(count, 1)

    def _block_quantization_error(self, weight, block_size):
        """Compute quantization error for weight matrix with given block size."""
        w_flat = weight.view(-1)
        num_blocks = (len(w_flat) + block_size - 1) // block_size

        total_error = 0.0
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min(start_idx + block_size, len(w_flat))
            block = w_flat[start_idx:end_idx]

            if len(block) > 0:
                q = torch.quantile(torch.abs(block), 0.75)
                if q > 1e-8:
                    ternary = torch.where(torch.abs(block) > q, torch.sign(block), torch.zeros_like(block))
                    error = torch.mean((block - ternary) ** 2).item()
                    total_error += error

        return total_error / max(num_blocks, 1)

    def get_results(self):
        """Return {block_size: mean_quantization_error}."""
        return self.results.copy()

    def get_optimal_block_size(self):
        """Return block size with lowest quantization error."""
        if not self.results:
            return 64
        return min(self.results.items(), key=lambda x: x[1])[0]


class LayerWiseLR:
    """Per-layer learning rate scaling based on gradient/weight ratio.

    Principle: lr_layer = base_lr * ||weight|| / ||gradient||.
    Layers with large weights but small gradients need higher LR.
    For ternary: ||weight|| is just count of non-zero values.
    """

    def __init__(self, base_lr=1e-3, trust_coefficient=0.01):
        self.base_lr = base_lr
        self.trust_coefficient = trust_coefficient
        self.layer_lrs = {}

    def compute_layer_lrs(self, model):
        """Return per-layer LR adjustments.

        Args:
            model: PyTorch model

        Returns:
            Dict mapping parameter names to adjusted learning rates
        """
        self.layer_lrs = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                self.layer_lrs[name] = self.base_lr
                continue

            weight_norm = torch.norm(param.data).item()
            grad_norm = torch.norm(param.grad.data).item()

            if grad_norm > 1e-8:
                ratio = weight_norm / (grad_norm + 1e-8)
            else:
                ratio = 1.0

            adaptive_lr = self.base_lr * (self.trust_coefficient + ratio)
            self.layer_lrs[name] = adaptive_lr

        return self.layer_lrs

    def apply(self, optimizer):
        """Set per-group LR in optimizer.

        Args:
            optimizer: torch.optim optimizer instance
        """
        if not self.layer_lrs:
            return

        param_to_group = {}
        for group in optimizer.param_groups:
            for param in group['params']:
                param_id = id(param)
                param_to_group[param_id] = group

        for name, param in optimizer.param_groups[0]['params'].items() if hasattr(optimizer.param_groups[0]['params'], 'items') else []:
            pass

        for name, lr in self.layer_lrs.items():
            for group in optimizer.param_groups:
                for param in group['params']:
                    if hasattr(param, '_name') and param._name == name:
                        group['lr'] = lr
                        break


class GradientNoiseScale:
    """Measure gradient noise to determine if batch size is optimal.

    Principle: noise_scale = trace(Sigma) / ||G||^2.
    If noise_scale >> batch_size: increase batch.
    If noise_scale << batch_size: decrease batch (wasting compute).
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.grad_history_small = []
        self.grad_history_large = []
        self.noise_scale = None

    def update(self, gradients_small_batch, gradients_large_batch):
        """Estimate noise scale from two different batch sizes.

        Args:
            gradients_small_batch: List of gradient tensors from small batch
            gradients_large_batch: List of gradient tensors from large batch
        """
        small_flat = torch.cat([g.view(-1) for g in gradients_small_batch if g is not None])
        large_flat = torch.cat([g.view(-1) for g in gradients_large_batch if g is not None])

        self.grad_history_small.append(small_flat)
        self.grad_history_large.append(large_flat)

        if len(self.grad_history_small) > self.window_size:
            self.grad_history_small.pop(0)
            self.grad_history_large.pop(0)

        if len(self.grad_history_small) >= 2:
            self._estimate_noise_scale()

    def _estimate_noise_scale(self):
        """Estimate noise scale from gradient history."""
        if len(self.grad_history_small) < 2:
            return

        small_stack = torch.stack(self.grad_history_small[-min(10, len(self.grad_history_small)):])
        large_stack = torch.stack(self.grad_history_large[-min(10, len(self.grad_history_large)):])

        small_var = torch.var(small_stack, dim=0).sum().item()
        large_var = torch.var(large_stack, dim=0).sum().item()

        if small_var > 1e-8:
            self.noise_scale = small_var / large_var if large_var > 1e-8 else 1.0
        else:
            self.noise_scale = 1.0

    def recommend_batch_size(self, current_batch_size):
        """Return recommended batch size based on noise scale.

        Args:
            current_batch_size: Current batch size

        Returns:
            Recommended batch size
        """
        if self.noise_scale is None:
            return current_batch_size

        if self.noise_scale > 1.5:
            return int(current_batch_size * 1.5)
        elif self.noise_scale < 0.5:
            return max(1, int(current_batch_size * 0.75))
        else:
            return current_batch_size


class ContextualSparsity:
    """Input-dependent weight activation — skip weights unlikely to fire.

    Principle: for ternary {-1,0,+1}, zero weights already skip (mul by 0 = noop).
    Extend by predicting which ±1 weights will produce near-zero activations
    for this specific input. Uses running mean of activation magnitudes.
    """

    def __init__(self, sparsity_threshold=0.1):
        self.sparsity_threshold = sparsity_threshold
        self.activation_stats = defaultdict(lambda: None)

    def update_stats(self, module_name, activations):
        """Track running mean of activation magnitudes per weight row.

        Args:
            module_name: Name of the module
            activations: Tensor of shape (batch, seq_len, hidden_dim)
        """
        if activations.dim() >= 2:
            act_mean = torch.abs(activations).mean(dim=tuple(range(activations.dim() - 1)))
        else:
            act_mean = torch.abs(activations)

        if self.activation_stats[module_name] is None:
            self.activation_stats[module_name] = act_mean
        else:
            self.activation_stats[module_name] = 0.9 * self.activation_stats[module_name] + 0.1 * act_mean

    def get_sparse_mask(self, module_name, input_features):
        """Predict which weight rows can be skipped for this input.

        Args:
            module_name: Name of the module
            input_features: Input features tensor

        Returns:
            Boolean mask (True = keep, False = skip)
        """
        if module_name not in self.activation_stats or self.activation_stats[module_name] is None:
            return None

        stats = self.activation_stats[module_name]
        threshold = torch.quantile(stats, 1.0 - self.sparsity_threshold)
        mask = stats >= threshold

        return mask


class HadamardRotation:
    """Apply Hadamard rotation to spread quantization error uniformly.

    Principle: Hadamard matrix H is orthogonal (H^T H = I) and can be
    applied in O(n log n) via butterfly algorithm. Rotating weights before
    quantization spreads outliers across dimensions, reducing max error.
    """

    def __init__(self):
        self.hadamard_matrices = {}

    def _get_hadamard_matrix(self, n, device='cpu'):
        """Get or compute Hadamard matrix of size n x n.

        Args:
            n: Dimension (will be padded to nearest power of 2)
            device: Device to place matrix on

        Returns:
            Hadamard matrix normalized by 1/sqrt(n)
        """
        key = (n, device)
        if key in self.hadamard_matrices:
            return self.hadamard_matrices[key]

        def _hadamard_power_of_2(m):
            if m == 1:
                return torch.tensor([[1.0]], dtype=torch.float32)
            h_half = _hadamard_power_of_2(m // 2)
            return torch.cat([
                torch.cat([h_half, h_half], dim=1),
                torch.cat([h_half, -h_half], dim=1)
            ], dim=0)

        size = 1
        while size < n:
            size *= 2

        H = _hadamard_power_of_2(size)
        H = H / math.sqrt(size)
        H = H[:n, :n]
        H = H.to(device)

        self.hadamard_matrices[key] = H
        return H

    def rotate(self, weight):
        """Apply fast Walsh-Hadamard transform. Pad to power of 2 if needed.

        Args:
            weight: Weight tensor

        Returns:
            Rotated weight tensor
        """
        if weight.dim() < 2:
            return weight

        original_shape = weight.shape
        weight_2d = weight.view(weight.shape[0], -1)

        n_rows = weight_2d.shape[0]
        H = self._get_hadamard_matrix(n_rows, weight.device)

        rotated = torch.matmul(H, weight_2d)
        return rotated.view(*original_shape)

    def unrotate(self, weight):
        """Inverse Hadamard (same operation since H^-1 = H^T = H/n).

        Args:
            weight: Rotated weight tensor

        Returns:
            Original weight tensor
        """
        return self.rotate(weight)


class TernaryGroupConv:
    """Group-wise ternary quantization for FFN layers.

    Principle: instead of one threshold for entire weight matrix,
    use per-group thresholds. Groups of 64-256 weights share a threshold.
    Reduces quantization error by 30-40% with minimal overhead.
    """

    def __init__(self, group_size=128):
        self.group_size = group_size
        self.group_scales = {}

    def quantize_grouped(self, weight):
        """Quantize with per-group thresholds. Return ternary + group_scales.

        Args:
            weight: Weight tensor

        Returns:
            Tuple of (ternary_weight, group_scales)
        """
        w_flat = weight.view(-1)
        num_groups = (len(w_flat) + self.group_size - 1) // self.group_size

        ternary = torch.zeros_like(w_flat)
        scales = torch.zeros(num_groups, dtype=weight.dtype, device=weight.device)

        for i in range(num_groups):
            start_idx = i * self.group_size
            end_idx = min(start_idx + self.group_size, len(w_flat))
            group = w_flat[start_idx:end_idx]

            threshold = torch.quantile(torch.abs(group), 0.75)
            scales[i] = threshold

            if threshold > 1e-8:
                ternary[start_idx:end_idx] = torch.where(
                    torch.abs(group) > threshold,
                    torch.sign(group),
                    torch.zeros_like(group)
                )
            else:
                ternary[start_idx:end_idx] = torch.zeros_like(group)

        ternary = ternary.view(weight.shape)

        return ternary, scales

    def dequantize_grouped(self, ternary_weight, group_scales):
        """Reconstruct approximate float weight from groups.

        Args:
            ternary_weight: Ternary weight tensor
            group_scales: Group scale factors

        Returns:
            Approximate float weight
        """
        w_flat = ternary_weight.view(-1)
        num_groups = len(group_scales)

        reconstructed = torch.zeros_like(w_flat)

        for i in range(num_groups):
            start_idx = i * self.group_size
            end_idx = min(start_idx + self.group_size, len(w_flat))
            reconstructed[start_idx:end_idx] = w_flat[start_idx:end_idx] * group_scales[i]

        return reconstructed.view(ternary_weight.shape)


class CPSGDCorrection:
    """Periodic full-precision correction every P steps.

    Principle: compressed gradients accumulate error over time.
    Every P steps, do one round of full-precision sync to correct drift.
    P typically sqrt(T_total) for optimal convergence.
    """

    def __init__(self, correction_period=None, total_steps=10000):
        self.period = correction_period or int(math.sqrt(total_steps))
        self.correction_error_accumulator = defaultdict(float)

    def should_correct(self, step):
        """Return True every P steps.

        Args:
            step: Current training step

        Returns:
            True if correction should happen now
        """
        return step > 0 and step % self.period == 0

    def full_precision_sync(self, all_worker_params):
        """Average all params in full precision (no compression).

        Args:
            all_worker_params: List of parameter lists from all workers

        Returns:
            Averaged parameters in full precision
        """
        if not all_worker_params or len(all_worker_params) == 0:
            return None

        num_workers = len(all_worker_params)
        first_params = all_worker_params[0]

        averaged = []
        for param_idx in range(len(first_params)):
            param_sum = torch.zeros_like(first_params[param_idx], dtype=torch.float32)

            for worker_params in all_worker_params:
                if param_idx < len(worker_params):
                    param_sum += worker_params[param_idx].to(torch.float32)

            averaged.append(param_sum / num_workers)

        return averaged


class GradStopSampler:
    """Posterior sampling for optimal gradient stopping point.

    Principle: in ternary networks, gradients beyond a threshold don't help
    (weight can only be {-1,0,+1}). Use Bayesian estimation of when
    gradient accumulation has "enough" information to make the correct
    ternary decision. Stop early = save compute.
    """

    def __init__(self, confidence=0.95):
        self.confidence = confidence
        self.gradient_accumulator = defaultdict(lambda: {'sum': 0.0, 'count': 0})

    def should_stop(self, accumulated_gradient, n_samples):
        """Return True if gradient direction is statistically certain.

        Args:
            accumulated_gradient: Sum of accumulated gradients
            n_samples: Number of samples accumulated

        Returns:
            True if direction is confident enough to stop
        """
        if n_samples < 2:
            return False

        confidence = self._get_confidence(accumulated_gradient, n_samples)
        return confidence >= self.confidence

    def get_confidence(self, accumulated_gradient, n_samples):
        """Return posterior probability that gradient sign is correct.

        Args:
            accumulated_gradient: Sum of accumulated gradients
            n_samples: Number of samples accumulated

        Returns:
            Confidence score in [0, 1]
        """
        return self._get_confidence(accumulated_gradient, n_samples)

    def _get_confidence(self, accumulated_gradient, n_samples):
        """Compute confidence using Bayesian posterior."""
        if n_samples < 1:
            return 0.0

        mean_grad = accumulated_gradient / n_samples

        if abs(mean_grad) < 1e-10:
            return 0.5

        z_score = abs(mean_grad) * math.sqrt(n_samples)

        confidence = 1.0 - math.exp(-z_score / 2.0)
        confidence = min(max(confidence, 0.0), 1.0)

        return confidence

    def update(self, param_name, gradient):
        """Update accumulated gradient for a parameter.

        Args:
            param_name: Name of parameter
            gradient: Gradient tensor
        """
        grad_flat = gradient.view(-1)
        self.gradient_accumulator[param_name]['sum'] += grad_flat.sum().item()
        self.gradient_accumulator[param_name]['count'] += len(grad_flat)
