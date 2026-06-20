"""GTF — Greedy Ternary Flipping for direct model modification.

Instead of LoRA adapters, GTF directly flips ternary weights {-1,0,+1}
based on gradient signals from new domain data. Modifies model in-place.

Protection against catastrophic forgetting:
- Never flip frozen/committed weights (from optimizer)
- Flip budget limits total modifications
- Gradient-based scoring ensures only beneficial flips
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class GTFAdapter:
    """Greedy Ternary Flipping for domain adaptation.

    Usage:
        gtf = GTFAdapter(model, optimizer=optimizer)
        gtf.evaluate_flips(data_loader, n_batches=10)
        stats = gtf.apply_flips(budget=0.001)
        print(stats)
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.ternary_modules = {}
        self.flip_scores = {}
        self.applied_flips = []

        for name, module in model.named_modules():
            if hasattr(module, 'quantize_enabled') and hasattr(module, 'group_size'):
                self.ternary_modules[name] = module

    def _get_ternary_values(self, module):
        """Get current ternary values and threshold for a module."""
        w = module.weight.data
        threshold = module.threshold_coeff * w.abs().mean()
        ternary = torch.zeros_like(w, dtype=torch.int8)
        ternary[w > threshold] = 1
        ternary[w < -threshold] = -1
        return ternary, threshold

    def evaluate_flips(self, data_loader, n_batches=10, device=None):
        """Score every possible flip using accumulated gradients.

        Uses first-order approximation: delta_loss ~ grad * delta_weight.
        Negative scores indicate beneficial flips.
        """
        self.flip_scores = {}

        frozen_masks = {}
        if self.optimizer is not None:
            frozen_masks = self.optimizer.get_frozen_mask()

        grad_accum = {}
        batches_seen = 0

        for input_ids, target_ids in data_loader:
            if batches_seen >= n_batches:
                break

            if device:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

            self.model.zero_grad()
            self.model.train()

            with torch.enable_grad():
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                loss.backward()

            for name, module in self.ternary_modules.items():
                if module.weight.grad is not None:
                    if name not in grad_accum:
                        grad_accum[name] = module.weight.grad.clone()
                    else:
                        grad_accum[name].add_(module.weight.grad)

            batches_seen += 1

        self.model.eval()

        for name, module in self.ternary_modules.items():
            if name not in grad_accum:
                continue

            avg_grad = grad_accum[name] / batches_seen
            ternary_vals, threshold = self._get_ternary_values(module)

            param_id = id(module.weight)
            is_frozen = frozen_masks.get(param_id, None)

            flat_ternary = ternary_vals.flatten()
            flat_grad = avg_grad.flatten()
            n_weights = flat_ternary.numel()

            if is_frozen is not None:
                flat_frozen = is_frozen.flatten()
            else:
                flat_frozen = torch.zeros(n_weights, dtype=torch.bool, device=flat_ternary.device)

            for target_val in [-1, 0, 1]:
                can_flip = (flat_ternary != target_val) & (~flat_frozen)
                if not can_flip.any():
                    continue

                indices = can_flip.nonzero(as_tuple=True)[0]
                current_vals = flat_ternary[indices].float()
                delta = float(target_val) - current_vals
                scores = flat_grad[indices] * delta

                for i in range(len(indices)):
                    idx_item = indices[i].item()
                    score = scores[i].item()
                    key = (name, idx_item)

                    if key not in self.flip_scores or score < self.flip_scores[key][0]:
                        self.flip_scores[key] = (
                            score,
                            flat_ternary[idx_item].item(),
                            target_val,
                        )

        return len(self.flip_scores)

    @torch.no_grad()
    def apply_flips(self, budget=0.001, min_score_improvement=0.0):
        """Apply top-scoring flips within budget.

        Args:
            budget: Fraction of total ternary params to flip (0.001 = 0.1%)
            min_score_improvement: Minimum negative score to apply
        """
        if not self.flip_scores:
            return {'applied': 0, 'reason': 'no_scores'}

        total_ternary = sum(m.weight.numel() for m in self.ternary_modules.values())
        max_flips = int(total_ternary * budget)

        sorted_flips = sorted(self.flip_scores.items(), key=lambda x: x[1][0])

        applied = 0
        flips_by_layer = {}

        for (name, flat_idx), (score, current_val, target_val) in sorted_flips:
            if applied >= max_flips:
                break
            if score >= -min_score_improvement:
                break

            module = self.ternary_modules[name]
            w = module.weight.data
            threshold = module.threshold_coeff * w.abs().mean()

            row = flat_idx // w.size(1)
            col = flat_idx % w.size(1)

            old_shadow = w[row, col].item()

            if target_val == 1:
                w[row, col] = threshold * 1.5
            elif target_val == -1:
                w[row, col] = -threshold * 1.5
            else:
                w[row, col] = 0.0

            self.applied_flips.append({
                'name': name, 'row': row, 'col': col,
                'old_shadow': old_shadow,
                'old_val': current_val, 'new_val': target_val,
                'score': score,
            })

            flips_by_layer[name] = flips_by_layer.get(name, 0) + 1
            applied += 1

        return {
            'applied': applied,
            'budget': max_flips,
            'total_ternary': total_ternary,
            'budget_pct': budget * 100,
            'avg_score': sum(f['score'] for f in self.applied_flips[-applied:]) / max(1, applied) if applied > 0 else 0,
            'flips_by_layer': flips_by_layer,
        }

    @torch.no_grad()
    def undo_flips(self, n=None):
        """Undo last n applied flips (or all)."""
        if n is None:
            n = len(self.applied_flips)

        undone = 0
        while self.applied_flips and undone < n:
            flip = self.applied_flips.pop()
            module = self.ternary_modules[flip['name']]
            module.weight.data[flip['row'], flip['col']] = flip['old_shadow']
            undone += 1

        return undone

    def get_stats(self):
        """Return summary statistics."""
        total_ternary = sum(m.weight.numel() for m in self.ternary_modules.values())
        return {
            'ternary_modules': len(self.ternary_modules),
            'total_ternary_params': total_ternary,
            'scored_flips': len(self.flip_scores),
            'applied_flips': len(self.applied_flips),
            'applied_pct': len(self.applied_flips) / max(1, total_ternary) * 100,
        }
