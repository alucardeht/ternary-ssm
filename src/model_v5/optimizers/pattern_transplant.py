import torch
import logging

logger = logging.getLogger(__name__)


class PatternTransplantation:
    """Transplant ternary patterns from committed to newly-committed weights.

    When a weight commits, copy structure from nearest neighbor: 80% donor +
    20% random. Cascades knowledge through layers.
    """

    def __init__(self, blend_ratio=0.8):
        self.blend_ratio = blend_ratio
        self.committed_patterns = {}
        self.transplant_count = 0

    def on_commit(self, param, param_name, ternary_pattern):
        """Called when weight commits. Transplant pattern from donor."""
        donor_name = self._find_donor(param_name)

        if donor_name is None:
            logger.debug(f"[Transplant] No donor for {param_name}")
            self.committed_patterns[param_name] = ternary_pattern.detach().clone()
            return

        donor_pattern = self.committed_patterns[donor_name]
        new_pattern = self._blend_patterns(donor_pattern, ternary_pattern,
                                           self.blend_ratio)

        with torch.no_grad():
            param.data = new_pattern.to(param.dtype)

        overlap = (new_pattern == donor_pattern).float().mean().item()
        logger.info(f"[Transplant] {param_name} <- {donor_name}, "
                   f"overlap={overlap:.1%}")

        self.committed_patterns[param_name] = new_pattern.detach().clone()
        self.transplant_count += 1

    def _find_donor(self, param_name):
        """Find committed donor with similar role."""
        parts = param_name.split('.')
        if len(parts) < 2:
            return None

        role_key = '.'.join(parts[-2:])

        candidates = [k for k in self.committed_patterns.keys()
                     if role_key in k]

        if not candidates:
            return None

        return candidates[-1]

    def _blend_patterns(self, donor, original, blend_ratio):
        """Blend donor (80%) with original (20%)."""
        if donor.shape != original.shape:
            return original

        mask = torch.rand_like(donor) < blend_ratio
        blended = torch.where(mask, donor, original)
        return blended

    def get_stats(self):
        return {
            'patterns_stored': len(self.committed_patterns),
            'transplants': self.transplant_count
        }
