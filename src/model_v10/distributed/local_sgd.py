"""Local SGD optimized for ternary weight networks.

Ternary weights {-1,0,+1} change slowly and discretely. Workers can train
independently for H=50-500 steps (vs H=1-10 for FP16) before synchronizing.

Synchronization uses majority vote for ternary params and averaging for FP16 params.
This exploits the discrete nature of ternary values: the most frequent value across
workers is the consensus, which is more robust than averaging.
"""

import torch


class TernaryLocalSGD:
    """Local SGD for ternary + FP16 hybrid networks.

    Discovers ternary vs FP16 parameters automatically by checking for
    quantize_enabled attribute on modules. Groups parameters by type and
    uses appropriate synchronization strategy for each.

    Args:
        model: PyTorch model with mixed ternary/FP16 parameters
        sync_interval: Synchronize every N steps (default: 100)
        ternary_sync: Synchronization method for ternary params (default: 'majority_vote')
        fp16_sync: Synchronization method for FP16 params (default: 'average')
    """

    def __init__(
        self,
        model,
        sync_interval=100,
        ternary_sync='majority_vote',
        fp16_sync='average',
    ):
        self.model = model
        self.sync_interval = sync_interval
        self.ternary_sync = ternary_sync
        self.fp16_sync = fp16_sync
        self.step_count = 0
        self.sync_count = 0

        self._discover_parameters()
        self._init_sync_history()

    def _discover_parameters(self):
        """Discover and group ternary vs FP16 parameters."""
        self.ternary_params = []
        self.fp16_params = []
        self.param_to_module = {}

        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if param is None:
                    continue

                self.param_to_module[id(param)] = module
                is_ternary = hasattr(module, 'quantize_enabled')
                if is_ternary:
                    self.ternary_params.append(param)
                else:
                    self.fp16_params.append(param)

    def _init_sync_history(self):
        """Initialize synchronization history tracking."""
        self.sync_history = {
            'sync_count': 0,
            'ternary_agreements': [],
            'ternary_flips': [],
            'fp16_changes': [],
        }

    def should_sync(self, step):
        """Check if synchronization should occur at this step."""
        return (step + 1) % self.sync_interval == 0

    def sync_ternary_params(self, all_worker_params):
        """Synchronize ternary parameters using majority vote.

        For each weight position:
        - Collect values across all workers
        - Count votes for -1, 0, +1
        - Winner = value with most votes
        - Update local weights to majority value
        - Soft reset shadow weights toward majority by setting threshold
          adjustment (shadow = 1.5 * majority_value for ±1, 0 for 0)

        Args:
            all_worker_params: List of param lists, one per worker.
                              Shape: (num_workers, num_ternary_params)
                              Each param should be shape (d1, d2, ...)

        Returns:
            sync_stats: Dict with 'agreement_pct' and 'num_flips'
        """
        if not self.ternary_params or not all_worker_params:
            return {'agreement_pct': 100.0, 'num_flips': 0}

        num_workers = len(all_worker_params)
        total_agreement = 0
        total_positions = 0
        total_flips = 0

        for i, local_param in enumerate(self.ternary_params):
            worker_values = []
            for worker_idx in range(num_workers):
                worker_param = all_worker_params[worker_idx][i]
                worker_values.append(worker_param.data)

            majority_value = self._compute_majority_vote(worker_values)

            positions_changed = (local_param.data != majority_value)
            num_flips = positions_changed.sum().item()
            total_flips += num_flips

            agreement = (~positions_changed).sum().item()
            total_agreement += agreement
            total_positions += local_param.numel()

            local_param.data.copy_(majority_value)

            if hasattr(local_param, '_shadow_weight'):
                threshold = 1.5
                local_param._shadow_weight[majority_value == 1] = threshold
                local_param._shadow_weight[majority_value == -1] = -threshold
                local_param._shadow_weight[majority_value == 0] = 0.0

        agreement_pct = (
            100.0 * total_agreement / total_positions
            if total_positions > 0
            else 100.0
        )

        return {
            'agreement_pct': agreement_pct,
            'num_flips': total_flips,
        }

    def sync_fp16_params(self, all_worker_params):
        """Synchronize FP16 parameters using averaging.

        For each parameter:
        - Average values across all workers
        - Update local parameter to average

        Args:
            all_worker_params: List of param lists, one per worker.
                              Each contains only FP16 parameters (already filtered).

        Returns:
            sync_stats: Dict with 'max_change' (max L2 distance of updates)
        """
        if not self.fp16_params or not all_worker_params:
            return {'max_change': 0.0}

        num_workers = len(all_worker_params)
        max_change = 0.0

        for i, local_param in enumerate(self.fp16_params):
            worker_values = []
            for worker_idx in range(num_workers):
                worker_param = all_worker_params[worker_idx][i]
                worker_values.append(worker_param.data)

            average_value = torch.stack(worker_values).mean(dim=0)

            change = (local_param.data - average_value).norm().item()
            max_change = max(max_change, change)

            local_param.data.copy_(average_value)

        return {'max_change': max_change}

    @staticmethod
    def _compute_majority_vote(values):
        """Compute majority vote across values.

        Args:
            values: List of tensors with ternary values {-1, 0, +1}

        Returns:
            majority: Tensor with same shape, containing majority value at each position
        """
        stacked = torch.stack(values)

        votes_pos = (stacked == 1).sum(dim=0)
        votes_zero = (stacked == 0).sum(dim=0)
        votes_neg = (stacked == -1).sum(dim=0)

        majority = torch.zeros_like(stacked[0])
        majority[votes_pos > votes_zero] = 1
        majority[votes_pos > votes_neg] = 1

        mask_zero = (votes_zero > votes_pos) & (votes_zero > votes_neg)
        majority[mask_zero] = 0

        mask_neg = (votes_neg > votes_pos) & (votes_neg > votes_zero)
        majority[mask_neg] = -1

        return majority

    def sync(self, all_worker_states):
        """Full synchronization: ternary via majority vote, FP16 via averaging.

        Args:
            all_worker_states: List of dicts, each containing:
                              - 'ternary_params': List of ternary parameters
                              - 'fp16_params': List of FP16 parameters

        Returns:
            full_stats: Dict combining ternary and FP16 sync statistics
        """
        self.step_count += 1

        if not self.should_sync(self.step_count):
            return None

        self.sync_count += 1

        all_ternary_params = [s['ternary_params'] for s in all_worker_states]
        all_fp16_params = [s['fp16_params'] for s in all_worker_states]

        ternary_stats = self.sync_ternary_params(all_ternary_params)
        fp16_stats = self.sync_fp16_params(all_fp16_params)

        self.sync_history['sync_count'] += 1
        self.sync_history['ternary_agreements'].append(ternary_stats['agreement_pct'])
        self.sync_history['ternary_flips'].append(ternary_stats['num_flips'])
        self.sync_history['fp16_changes'].append(fp16_stats['max_change'])

        full_stats = {
            'sync_step': self.sync_count,
            'ternary': ternary_stats,
            'fp16': fp16_stats,
        }

        return full_stats

    def get_sync_stats(self):
        """Return statistics about synchronization history.

        Returns:
            dict: Statistics including agreement percentages, flip counts, etc.
        """
        if not self.sync_history['ternary_agreements']:
            return {
                'total_syncs': 0,
                'ternary_avg_agreement_pct': 0.0,
                'ternary_total_flips': 0,
                'fp16_max_change': 0.0,
            }

        return {
            'total_syncs': self.sync_history['sync_count'],
            'ternary_avg_agreement_pct': (
                sum(self.sync_history['ternary_agreements'])
                / len(self.sync_history['ternary_agreements'])
            ),
            'ternary_total_flips': sum(self.sync_history['ternary_flips']),
            'fp16_max_change': max(self.sync_history['fp16_changes'])
            if self.sync_history['fp16_changes']
            else 0.0,
        }

    def step(self):
        """Increment step counter (call after each forward/backward/optimizer step)."""
        self.step_count += 1
