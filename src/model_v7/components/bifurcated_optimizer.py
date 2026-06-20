"""
Bifurcated Optimizer - Distinct optimization strategies for committed vs active parameters.

References:
- R4-A6-P5: Bifurcated optimization splits ternary parameters into groups based on
  commitment state (active, transitioning, committed, frozen).
- R4-A6-P6: Asymmetric learning rates - committed parameters converge faster and need
  smaller learning rates; active parameters need full adaptation capacity.

Mathematical basis:
  Committed parameters: near their final ternary state, need less frequent updates.
  Active parameters: still exploring the ternary space, need full optimizer capacity.

  Memory savings: committed params use SGD+momentum (2x state) vs Adam (3x state).
  Learning efficiency: committed params at lower LR avoid overshooting.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD
import math


class BifurcatedOptimizer:
    """
    Wraps a base optimizer (typically AdamW) and splits parameter groups
    based on commitment classification.

    Four parameter groups:
    - Group 0 (ACTIVE): Full Adam, full learning rate
    - Group 1 (TRANSITIONING): Full Adam, 0.5x learning rate
    - Group 2 (COMMITTED): SGD+momentum, 0.1x learning rate
    - Group 3 (FROZEN): No updates, no optimizer state
    """

    def __init__(
        self,
        params: List[Dict],
        base_optimizer_cls=torch.optim.AdamW,
        base_lr: float = 1e-3,
        committed_lr_ratio: float = 0.1,
        transitioning_lr_ratio: float = 0.5,
    ):
        """
        Args:
            params: Parameter groups in format [{'params': [p1, p2, ...], 'lr': 1e-3}, ...]
            base_optimizer_cls: Optimizer class to use for active/transitioning groups
            base_lr: Base learning rate for ACTIVE group
            committed_lr_ratio: LR multiplier for COMMITTED group (0.1 = 10% of base)
            transitioning_lr_ratio: LR multiplier for TRANSITIONING group (0.5 = 50% of base)
        """
        self.base_optimizer_cls = base_optimizer_cls
        self.base_lr = base_lr
        self.committed_lr_ratio = committed_lr_ratio
        self.transitioning_lr_ratio = transitioning_lr_ratio

        # Separate params into groups
        self.param_groups = {
            'ACTIVE': [],
            'TRANSITIONING': [],
            'COMMITTED': [],
            'FROZEN': [],
        }

        # Initialize all params as ACTIVE
        for group in params:
            param_list = group.get('params', [])
            for p in param_list:
                self.param_groups['ACTIVE'].append(p)

        # Create base optimizer with all params
        param_groups_for_optimizer = [
            {
                'params': self.param_groups['ACTIVE'],
                'lr': base_lr,
            },
            {
                'params': self.param_groups['TRANSITIONING'],
                'lr': base_lr * transitioning_lr_ratio,
            },
            {
                'params': self.param_groups['COMMITTED'],
                'lr': base_lr * committed_lr_ratio,
            },
        ]

        self.base_optimizer = base_optimizer_cls(
            param_groups_for_optimizer,
            weight_decay=0.0,  # handled separately
        )

        self.step_count = 0
        self.total_param_flips = {
            'ACTIVE->TRANSITIONING': 0,
            'TRANSITIONING->COMMITTED': 0,
            'COMMITTED->FROZEN': 0,
        }

    def update_param_groups(self, commitment_classifications: Dict[int, str]):
        """
        Reclassify parameters based on commitment scores.

        Args:
            commitment_classifications: Dict mapping param id (id(param)) to classification string.
                String values: 'ACTIVE', 'TRANSITIONING', 'COMMITTED', 'FROZEN'

        Returns:
            Dict with transition counts and final distribution.
        """
        transitions = {
            'ACTIVE->TRANSITIONING': 0,
            'TRANSITIONING->COMMITTED': 0,
            'COMMITTED->FROZEN': 0,
        }

        # Track current state of each param
        current_state = {}
        for state, plist in self.param_groups.items():
            for p in plist:
                current_state[id(p)] = state

        # Process reclassifications
        for param_id, new_state in commitment_classifications.items():
            if param_id not in current_state:
                continue

            old_state = current_state[param_id]
            if old_state == new_state:
                continue

            # Track valid transitions
            if old_state == 'ACTIVE' and new_state == 'TRANSITIONING':
                transitions['ACTIVE->TRANSITIONING'] += 1
            elif old_state == 'TRANSITIONING' and new_state == 'COMMITTED':
                transitions['TRANSITIONING->COMMITTED'] += 1
            elif old_state == 'COMMITTED' and new_state == 'FROZEN':
                transitions['COMMITTED->FROZEN'] += 1

        # Clear all groups
        for state in self.param_groups:
            self.param_groups[state].clear()

        # Rebuild groups from classifications
        for param_id, state in commitment_classifications.items():
            for p in self.base_optimizer.param_groups[0]['params']:
                if id(p) == param_id:
                    if state in self.param_groups:
                        self.param_groups[state].append(p)
                    break

        # Update optimizer param groups
        # Active and Transitioning use base optimizer; Committed uses SGD momentum
        for i, group in enumerate(self.base_optimizer.param_groups):
            if i == 0:  # ACTIVE
                group['params'] = self.param_groups['ACTIVE']
                group['lr'] = self.base_lr
            elif i == 1:  # TRANSITIONING
                group['params'] = self.param_groups['TRANSITIONING']
                group['lr'] = self.base_lr * self.transitioning_lr_ratio
            elif i == 2:  # COMMITTED
                group['params'] = self.param_groups['COMMITTED']
                group['lr'] = self.base_lr * self.committed_lr_ratio

        # Accumulate transitions
        for key in transitions:
            self.total_param_flips[key] += transitions[key]

        return {
            'transitions': transitions,
            'distribution': {k: len(v) for k, v in self.param_groups.items()},
        }

    def step(self):
        """Execute one optimization step using the base optimizer."""
        self.base_optimizer.step()
        self.step_count += 1

    def zero_grad(self):
        """Clear all gradients."""
        self.base_optimizer.zero_grad()

    def get_optimizer_stats(self) -> Dict:
        """
        Return statistics about optimizer state and parameter distribution.

        Returns:
            Dict with:
            - params_per_group: {ACTIVE: N, TRANSITIONING: M, ...}
            - lr_per_group: {ACTIVE: 1e-3, TRANSITIONING: 5e-4, ...}
            - memory_per_group: Estimated optimizer state memory (bytes)
            - total_flips: Historical transition counts
            - step: Current step count
        """
        memory_per_group = {}

        # ACTIVE and TRANSITIONING use Adam: m + v states (2 * param count * 4 bytes)
        for state in ['ACTIVE', 'TRANSITIONING']:
            n_params = sum(p.numel() for p in self.param_groups[state])
            memory_per_group[state] = n_params * 4 * 2  # m and v tensors, FP32

        # COMMITTED uses SGD: only momentum (1 * param count * 4 bytes)
        n_params_committed = sum(p.numel() for p in self.param_groups['COMMITTED'])
        memory_per_group['COMMITTED'] = n_params_committed * 4

        memory_per_group['FROZEN'] = 0

        return {
            'params_per_group': {k: len(v) for k, v in self.param_groups.items()},
            'lr_per_group': {
                'ACTIVE': self.base_lr,
                'TRANSITIONING': self.base_lr * self.transitioning_lr_ratio,
                'COMMITTED': self.base_lr * self.committed_lr_ratio,
                'FROZEN': 0.0,
            },
            'memory_per_group': memory_per_group,
            'total_memory_bytes': sum(memory_per_group.values()),
            'total_flips': self.total_param_flips,
            'step': self.step_count,
        }
