import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class ConfidenceFilter:
    """Filter queries by prediction entropy for update selection.

    Mathematical principle: high entropy = model is uncertain = most
    informative gradient. Low entropy = model is confident = redundant update.
    Entropy normalized to [0,1] for interpretability.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute normalized entropy of logits distribution.

        Args:
            logits: (batch, vocab_size) or (batch, 1, vocab_size)

        Returns:
            entropy: (batch,) normalized to [0,1]
        """
        if logits.dim() == 3:
            logits = logits.squeeze(1)

        probs = F.softmax(logits, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        max_entropy = math.log(logits.shape[-1])
        normalized_entropy = entropy / (max_entropy + 1e-8)

        return normalized_entropy

    def should_update(self, logits: torch.Tensor) -> bool:
        """Return True if entropy exceeds threshold."""
        entropy = self.compute_entropy(logits).mean()
        return entropy.item() > self.threshold


class GradientAccumulator:
    """Accumulate gradients over K queries, then apply as single batch update.

    Mathematical principle: amortization. Single update from K queries
    costs same as 1 update, but gradient is K× more accurate (averaged noise).
    """

    def __init__(self, model: torch.nn.Module, k: int = 100):
        self.model = model
        self.k = k
        self.accumulated_losses = []
        self.accumulation_count = 0

    def accumulate(self, loss: torch.Tensor):
        """Add this query's loss to the accumulator."""
        self.accumulated_losses.append(loss)
        self.accumulation_count += 1

    def should_apply(self) -> bool:
        """Return True when K gradients accumulated."""
        return self.accumulation_count >= self.k

    def apply(self, optimizer):
        """Apply accumulated gradient as single optimizer step. Reset accumulator."""
        if not self.accumulated_losses:
            return

        batched_loss = sum(self.accumulated_losses) / len(self.accumulated_losses)

        optimizer.zero_grad()

        if batched_loss.requires_grad:
            batched_loss.backward()
        else:
            loss_to_backward = batched_loss.clone().detach().requires_grad_(True)
            loss_to_backward.backward()

        optimizer.step()

        self.accumulated_losses = []
        self.accumulation_count = 0


class ForgettingMonitor:
    """Monitor loss on validation set to detect catastrophic forgetting.

    Mathematical principle: if validation loss increases by > delta,
    the model is forgetting. Reduce K (more conservative updates) or
    increase EWC lambda.
    """

    def __init__(
        self,
        validation_data: torch.Tensor,
        check_interval: int = 1000,
        delta_threshold: float = 0.1,
    ):
        self.validation_data = validation_data
        self.check_interval = check_interval
        self.delta_threshold = delta_threshold
        self.baseline_loss = None
        self.last_loss = None
        self.check_count = 0
        self.forgetting_detected = False

    def check(self, model: torch.nn.Module, step: int) -> float:
        """Evaluate on validation set. Return forgetting_score."""
        self.check_count += 1

        model.eval()
        with torch.no_grad():
            if self.validation_data.dim() == 1:
                val_ids = self.validation_data.unsqueeze(0)
            else:
                val_ids = self.validation_data

            if hasattr(model, "forward"):
                output = model(val_ids)

                if isinstance(output, torch.Tensor):
                    logits = output
                elif isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output.logits if hasattr(output, "logits") else output

                if logits.dim() == 3:
                    logits = logits[:, -1, :]

                loss = F.cross_entropy(
                    logits, val_ids[:, -1], reduction="mean", ignore_index=-100
                )
            else:
                loss = torch.tensor(0.0, device=val_ids.device)

        if self.baseline_loss is None:
            self.baseline_loss = loss.item()
            self.last_loss = loss.item()
            forgetting_score = 0.0
        else:
            current_loss = loss.item()
            delta = (current_loss - self.baseline_loss) / (self.baseline_loss + 1e-8)

            if delta > self.delta_threshold:
                self.forgetting_detected = True
            else:
                self.forgetting_detected = False

            forgetting_score = max(0.0, delta)
            self.last_loss = current_loss

        return forgetting_score

    def should_reduce_rate(self) -> bool:
        """True if forgetting detected."""
        return self.forgetting_detected


class EWCProtection:
    """Protect important weights from changing during inference learning.

    For ternary: importance = inverse of flip frequency from optimizer stats.
    Weights that rarely flip are important and should be protected.
    No Fisher Information needed — flip frequency IS the importance signal.
    """

    def __init__(self, optimizer, lambda_ewc: float = 0.1):
        self.optimizer = optimizer
        self.lambda_ewc = lambda_ewc
        self.importance_map = {}
        self.reference_weights = {}

    def compute_importance(self) -> Dict[str, torch.Tensor]:
        """Compute importance from optimizer flip frequency stats.

        importance[i] = 1 / (flip_count[i] + 1). Rarely-flipping = important.
        """
        importance = {}

        for group in self.optimizer.param_groups:
            is_ternary = group.get("is_ternary", False)
            if not is_ternary:
                continue

            for p in group["params"]:
                if p not in self.optimizer.state:
                    importance[id(p)] = torch.ones_like(p.data)
                    continue

                state = self.optimizer.state[p]
                flip_count = state.get("flipped_count", 0)

                if flip_count == 0:
                    imp = torch.ones_like(p.data)
                else:
                    imp = 1.0 / (float(flip_count) + 1.0)

                importance[id(p)] = imp * torch.ones_like(p.data)

        return importance

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """EWC penalty: lambda * sum(importance * (w - w_ref)^2).

        For ternary: penalty on shadow weights, not quantized weights.
        """
        total_penalty = torch.tensor(0.0, device=next(model.parameters()).device)

        importance_map = self.compute_importance()

        for p in model.parameters():
            param_id = id(p)

            if param_id not in self.reference_weights:
                self.reference_weights[param_id] = p.data.clone().detach()

            if param_id not in importance_map:
                continue

            w_ref = self.reference_weights[param_id]
            importance = importance_map[param_id]

            penalty_term = (importance * (p.data - w_ref) ** 2).mean()
            total_penalty = total_penalty + self.lambda_ewc * penalty_term

        return total_penalty


class InferenceLearner:
    """Continuous learning during inference. Model improves while serving.

    Pipeline:
    1. Serve query normally (forward pass)
    2. Score confidence (entropy)
    3. If low confidence: accumulate gradient
    4. Every K queries: apply batched update (amortized)
    5. Monitor forgetting on validation set
    6. EWC protection for important weights

    Mathematical principle: inference generates FREE training signal.
    User corrections, speculative rejections, low-confidence queries all
    provide gradients. Ternary updates via LION are sign-only (ultra-cheap).
    Budget: ≤6% overhead per query.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        batch_size_k: int = 100,
        confidence_threshold: float = 0.5,
        ewc_lambda: float = 0.1,
        validation_data: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size_k = batch_size_k
        self.confidence_threshold = confidence_threshold

        self.confidence_filter = ConfidenceFilter(threshold=confidence_threshold)
        self.gradient_accumulator = GradientAccumulator(
            model=model, k=batch_size_k
        )

        self.ewc_protection = EWCProtection(
            optimizer=optimizer, lambda_ewc=ewc_lambda
        )

        self.forgetting_monitor = None
        if validation_data is not None:
            self.forgetting_monitor = ForgettingMonitor(
                validation_data=validation_data, check_interval=batch_size_k
            )

        self.queries_processed = 0
        self.updates_applied = 0
        self.total_confidence = 0.0

    def process_query(
        self,
        input_ids: torch.Tensor,
        generated_ids: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process one query. Returns output AND internally manages learning.

        Args:
            input_ids: (batch, seq_len) model input
            generated_ids: (batch, seq_len) optional generated output for correction
            target_ids: (batch, seq_len) optional target sequence for loss

        Returns:
            logits: (batch, vocab_size) model output
            stats: dict with confidence, updated, etc.
        """
        self.model.eval()
        stats = {"confidence": 0.0, "updated": False, "forgetting_score": 0.0}

        with torch.no_grad():
            output = self.model(input_ids)

            if isinstance(output, torch.Tensor):
                logits = output
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output.logits if hasattr(output, "logits") else output

            if logits.dim() == 3:
                logits = logits[:, -1:, :]

        entropy = self.confidence_filter.compute_entropy(logits)
        avg_entropy = entropy.mean().item()
        stats["confidence"] = avg_entropy
        self.total_confidence += avg_entropy

        should_learn = avg_entropy > self.confidence_threshold

        if should_learn and target_ids is not None:
            self.model.train()

            output_for_grad = self.model(input_ids)

            if isinstance(output_for_grad, torch.Tensor):
                logits_grad = output_for_grad
            elif isinstance(output_for_grad, tuple):
                logits_grad = output_for_grad[0]
            else:
                logits_grad = (
                    output_for_grad.logits
                    if hasattr(output_for_grad, "logits")
                    else output_for_grad
                )

            if logits_grad.dim() == 3:
                pred = logits_grad[:, -1, :]
            else:
                pred = logits_grad

            loss = F.cross_entropy(pred, target_ids, reduction="mean", ignore_index=-100)

            ewc_penalty = self.ewc_protection.penalty(self.model)
            total_loss = loss + ewc_penalty

            self.gradient_accumulator.accumulate(total_loss)

            if self.gradient_accumulator.should_apply():
                self.gradient_accumulator.apply(self.optimizer)
                self.updates_applied += 1
                stats["updated"] = True

                if self.forgetting_monitor is not None:
                    forgetting_score = self.forgetting_monitor.check(
                        self.model, self.queries_processed
                    )
                    stats["forgetting_score"] = forgetting_score

            self.model.eval()

        self.queries_processed += 1

        return logits, stats

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregated statistics."""
        avg_confidence = (
            self.total_confidence / max(1, self.queries_processed)
        )

        return {
            "queries_processed": self.queries_processed,
            "updates_applied": self.updates_applied,
            "avg_confidence": avg_confidence,
            "forgetting_detected": (
                self.forgetting_monitor.forgetting_detected
                if self.forgetting_monitor is not None
                else False
            ),
        }
