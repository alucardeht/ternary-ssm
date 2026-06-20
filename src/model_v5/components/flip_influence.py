"""Flip Influence Decomposition (R4-A4-M8).

Per-flip loss attribution: I_i = g_i * delta_i.
Identifies culprit weights for loss spikes.
Enables surgical rollback of specific harmful weights.
"""
import torch
import torch.nn as nn
from typing import Optional


class FlipInfluenceTracker:
    """Track per-flip influence on loss for surgical rollback."""

    def __init__(self, top_k: int = 100):
        """
        Args:
            top_k: track top-k most influential flips per step
        """
        self.top_k = top_k
        self._influence_history = []  # (step, [(name, pos, influence, delta)])
        self._harmful_flips = []      # flips that caused loss increase

    def compute_influences(self, model: nn.Module,
                            step_flips: dict, step: int) -> dict:
        """Compute influence of each flip using current gradients.

        I_i = grad_i * delta_i
        For ternary: delta in {-2, -1, +1, +2}, so influence is exact.

        Args:
            model: model with gradients computed
            step_flips: dict from DeltaChronicle.record_step()

        Returns:
            dict with top influential flips
        """
        all_influences = []

        for name, p in model.named_parameters():
            if name not in step_flips or p.grad is None:
                continue

            flips = step_flips[name]
            positions = flips['positions']
            old_vals = flips['old'].float()
            new_vals = flips['new'].float()
            deltas = new_vals - old_vals

            grad = p.grad.data.flatten()
            flat_positions = positions.flatten() if positions.dim() == 1 else \
                             positions[:, 0] if positions.shape[1] == 1 else \
                             sum(positions[:, i] * p.shape[i+1:].numel()
                                 for i in range(positions.shape[1]))

            flat_positions = flat_positions.clamp(0, len(grad) - 1).long()
            flip_grads = grad[flat_positions]

            influences = flip_grads * deltas
            for i in range(len(influences)):
                all_influences.append({
                    'name': name,
                    'position': positions[i].tolist() if positions.dim() > 1 else positions[i].item(),
                    'influence': influences[i].item(),
                    'delta': deltas[i].item(),
                    'grad': flip_grads[i].item(),
                })

        all_influences.sort(key=lambda x: abs(x['influence']), reverse=True)
        top = all_influences[:self.top_k]

        self._influence_history.append((step, top))

        harmful = [f for f in top if f['influence'] > 0]
        if harmful:
            self._harmful_flips.extend([(step, f) for f in harmful[:10]])

        return {
            'total_flips': len(all_influences),
            'top_influence': top[0]['influence'] if top else 0,
            'harmful_count': len(harmful),
            'beneficial_count': len([f for f in top if f['influence'] < 0]),
        }

    def get_rollback_candidates(self, loss_spike_threshold: float = 0.5) -> list:
        """Get flips that should be rolled back to fix a loss spike.

        Returns list of (param_name, position, current_value, recommended_value).
        """
        candidates = []
        for step, flips in self._harmful_flips[-50:]:
            for f in flips if isinstance(flips, list) else [flips]:
                if abs(f['influence']) > loss_spike_threshold:
                    old_delta = f['delta']
                    candidates.append({
                        'name': f['name'],
                        'position': f['position'],
                        'revert_delta': -old_delta,
                        'influence': f['influence'],
                        'step': step,
                    })
        return sorted(candidates, key=lambda x: abs(x['influence']), reverse=True)

    def surgical_rollback(self, model: nn.Module, candidates: list,
                          max_rollbacks: int = 10) -> int:
        """Apply surgical rollback of specific harmful flips.

        Returns number of weights rolled back.
        """
        rolled_back = 0
        for c in candidates[:max_rollbacks]:
            for name, p in model.named_parameters():
                if name == c['name']:
                    flat = p.data.flatten()
                    pos = c['position']
                    if isinstance(pos, list):
                        idx = 0
                        for i, pi in enumerate(pos):
                            idx = idx * p.shape[i + 1] if i < len(pos) - 1 else idx + pi
                        pos = idx
                    if 0 <= pos < len(flat):
                        flat[pos] += c['revert_delta']
                        flat[pos] = flat[pos].clamp(-1, 1).round()
                        p.data = flat.reshape(p.shape)
                        rolled_back += 1
                    break
        return rolled_back

    def get_stats(self) -> dict:
        return {
            'history_entries': len(self._influence_history),
            'total_harmful_flips': len(self._harmful_flips),
        }
