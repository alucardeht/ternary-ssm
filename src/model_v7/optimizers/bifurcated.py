import torch
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


class BifurcatedOptimizer:
    """Separate optimizers for committed (SGD) vs uncommitted (Adam) ternary weights.

    Math: Committed weights have low-curvature landscape (fine-tuning),
    uncommitted have high-curvature (exploration). Different optimizers per regime.
    """

    def __init__(self, params, base_lr=6e-4, commitment_threshold=0.01):
        self.params = list(params)
        self.base_lr = base_lr
        self.threshold = commitment_threshold

        self.crystal = []
        self.liquid = []
        self.steps = 0

        self.sgd = optim.SGD([], lr=base_lr*0.2, momentum=0.9)
        self.adam = optim.AdamW([], lr=base_lr*2.5, betas=(0.9, 0.999))

        from ..components.ternary import CommitmentMonitor
        self.detector = CommitmentMonitor(check_interval=100)

    def step(self, closure=None):
        if self.steps % 50 == 0:
            self._repartition()

        if self.sgd.param_groups:
            self.sgd.step()
        if self.adam.param_groups:
            self.adam.step()

        self.steps += 1

    def zero_grad(self):
        self.sgd.zero_grad()
        self.adam.zero_grad()

    def _repartition(self):
        new_crystal, new_liquid = [], []

        for p in self.params:
            if not hasattr(p, 'ternary'):
                continue

            committed_mask, _ = self.detector.margin_committed_mask(p)
            is_committed = committed_mask.float().mean().item() > 0.7

            if is_committed:
                new_crystal.append(p)
                if not any(p in pg['params'] for pg in self.sgd.param_groups):
                    self.sgd.add_param_group({'params': [p]})
            else:
                new_liquid.append(p)
                if not any(p in pg['params'] for pg in self.adam.param_groups):
                    self.adam.add_param_group({'params': [p]})

        self.crystal = new_crystal
        self.liquid = new_liquid

        if self.steps % 100 == 0:
            logger.info(f"[Bifurcated] Crystal={len(new_crystal)}, Liquid={len(new_liquid)}")

    @property
    def param_groups(self):
        return self.sgd.param_groups + self.adam.param_groups

    @property
    def state(self):
        combined = {}
        combined.update(self.sgd.state)
        combined.update(self.adam.state)
        return combined
