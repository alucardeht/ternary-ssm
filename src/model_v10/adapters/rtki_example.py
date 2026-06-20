"""RTKI Usage Examples — Production patterns for domain adaptation.

This file demonstrates practical use of RTKI for:
1. Domain adaptation with quality monitoring
2. Periodic residual tracking during training
3. Inference-time quality recovery
"""

import torch
from .rtki import RTKIAdapter


class DomainAdaptationPipeline:
    """Complete pipeline for RTKI-based domain adaptation."""

    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.rtki = RTKIAdapter(
            model,
            optimizer,
            top_k_fraction=0.01,
            residual_weight=0.3
        )

    def adapt_to_domain(self, data_loader, n_adaptation_cycles=5, budget=0.001):
        """Adapt model to new domain using RTKI-guided flipping.

        Workflow:
          1. For each adaptation cycle:
             - Evaluate which weights to flip using gradient + residual signals
             - Apply top-k flips within budget
             - Track quality metrics
          2. Report adaptation stats
        """
        adaptation_history = []

        for cycle in range(n_adaptation_cycles):
            stats = self.rtki.evaluate_and_flip(
                data_loader,
                budget=budget,
                n_batches=10,
                device=self.device
            )

            quality = self.rtki.get_stats()['quality_map']

            cycle_report = {
                'cycle': cycle,
                'flips_applied': stats['applied'],
                'avg_score': stats.get('avg_combined_score', 0),
                'quality_map': quality,
            }
            adaptation_history.append(cycle_report)

            print(f"Cycle {cycle}: Applied {stats['applied']} flips, "
                  f"avg quality: {sum(quality.values())/len(quality):.6f}")

        return adaptation_history


class TrainingWithResidualMonitoring:
    """Training loop with periodic residual quality monitoring."""

    def __init__(self, model, optimizer, rtki_adapter):
        self.model = model
        self.optimizer = optimizer
        self.rtki = rtki_adapter
        self.quality_history = []

    def train_step(self, input_ids, target_ids):
        """Single training step."""
        self.model.zero_grad()
        logits = self.model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, data_loader, total_steps, monitoring_interval=1000):
        """Train with periodic residual quality monitoring.

        Every monitoring_interval steps:
          - Update residual tracker
          - Log quality metrics
          - Detect quantization degradation
        """
        step = 0

        for epoch in range(total_steps // len(data_loader) + 1):
            for input_ids, target_ids in data_loader:
                loss = self.train_step(input_ids, target_ids)

                if step % monitoring_interval == 0 and step > 0:
                    self.rtki.update_residuals()
                    quality = self.rtki.tracker.get_quality_map()
                    self.quality_history.append({
                        'step': step,
                        'loss': loss,
                        'quality': quality,
                    })

                    avg_quality = sum(quality.values()) / len(quality)
                    print(f"Step {step}: loss={loss:.4f}, "
                          f"avg_residual_quality={avg_quality:.6f}")

                step += 1
                if step >= total_steps:
                    break

            if step >= total_steps:
                break

        return self.quality_history


class InferenceWithQualityRecovery:
    """Inference using sparse residual corrections for quality recovery."""

    def __init__(self, model, rtki_adapter, correction_alpha=0.1):
        self.model = model
        self.rtki = rtki_adapter
        self.correction_alpha = correction_alpha

    def infer_with_recovery(self, input_ids):
        """Run inference with sparse residual corrections applied.

        Residual corrections recover some of the information lost during
        ternary quantization, improving output quality at modest cost.
        """
        with self.rtki.tracker.apply_corrections(alpha=self.correction_alpha):
            logits = self.model(input_ids)
            return logits

    def infer_without_recovery(self, input_ids):
        """Run inference in pure ternary mode (no corrections)."""
        logits = self.model(input_ids)
        return logits

    def compare_modes(self, input_ids, target_ids):
        """Compare output quality with and without residual corrections."""
        with torch.no_grad():
            logits_with = self.infer_with_recovery(input_ids)
            logits_without = self.infer_without_recovery(input_ids)

            loss_with = torch.nn.functional.cross_entropy(
                logits_with.view(-1, logits_with.size(-1)),
                target_ids.view(-1)
            ).item()

            loss_without = torch.nn.functional.cross_entropy(
                logits_without.view(-1, logits_without.size(-1)),
                target_ids.view(-1)
            ).item()

            improvement = (loss_without - loss_with) / loss_without * 100
            return {
                'loss_without': loss_without,
                'loss_with': loss_with,
                'improvement_pct': improvement,
            }


class ProgressiveAdaptation:
    """Multi-stage adaptation with increasing flip budget."""

    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.rtki = RTKIAdapter(model, optimizer)
        self.stage = 0

    def stage_1_conservative(self, data_loader):
        """Stage 1: Conservative adaptation (0.05% flips)."""
        self.stage = 1
        return self.rtki.evaluate_and_flip(
            data_loader,
            budget=0.0005,
            n_batches=5,
            device=self.device
        )

    def stage_2_balanced(self, data_loader):
        """Stage 2: Balanced adaptation (0.1% flips)."""
        self.stage = 2
        return self.rtki.evaluate_and_flip(
            data_loader,
            budget=0.001,
            n_batches=10,
            device=self.device
        )

    def stage_3_aggressive(self, data_loader):
        """Stage 3: Aggressive adaptation (0.2% flips)."""
        self.stage = 3
        return self.rtki.evaluate_and_flip(
            data_loader,
            budget=0.002,
            n_batches=15,
            device=self.device
        )

    def run_full_pipeline(self, data_loaders, device='cuda'):
        """Run all 3 adaptation stages sequentially.

        Args:
            data_loaders: Dict of {stage_name: data_loader}
                         Expected: {'conservative': dl1, 'balanced': dl2, 'aggressive': dl3}
        """
        results = {}

        results['stage_1'] = self.stage_1_conservative(data_loaders.get('conservative'))
        results['stage_2'] = self.stage_2_balanced(data_loaders.get('balanced'))
        results['stage_3'] = self.stage_3_aggressive(data_loaders.get('aggressive'))

        return results
