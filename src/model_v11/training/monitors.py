import torch
import torch.nn as nn
import json
from typing import Callable


_global_vote_optimizer = None


def get_global_vote_optimizer():
    """Get or create global vote-buffer optimizer."""
    global _global_vote_optimizer
    if _global_vote_optimizer is None:
        from ..components.ternary import VoteBufferOptimizer

        _global_vote_optimizer = VoteBufferOptimizer(
            vote_window=50,
            k_sigma=3.0,
            variance_ema_beta=0.99,
        )
    return _global_vote_optimizer


class TrainingMonitor:
    """Tracks training metrics for analysis."""

    def __init__(self):
        self.history = {
            "step": [],
            "loss": [],
            "lr": [],
            "tokens_per_sec": [],
            "grad_norm": [],
            "epoch": [],
        }
        self.router_history = {}
        self.ternary_history = {}

    def log(self, step: int, **kwargs):
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def print_status(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_per_sec: float,
        grad_norm: float,
        extra: str = "",
    ):
        print(
            f"Step {step:6d} | Loss {loss:.4f} | LR {lr:.2e} | "
            f"Tok/s {tokens_per_sec:.0f} | GradNorm {grad_norm:.4f}{extra}"
        )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                _m2_shadow_dtype = torch.bfloat16 if ('embed' in name or 'embedding' in name) else torch.float32
                self.shadow[name] = param.data.clone().to(_m2_shadow_dtype)

        self._m1_evicted: set = set()
        self._m1_name_to_module: dict = {}
        for name, module in model.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                full_name = f"{name}.{pname}" if name else pname
                self._m1_name_to_module[full_name] = module

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue

            if name not in self._m1_evicted and name in self.shadow:
                _m1_mod = self._m1_name_to_module.get(name)
                if (_m1_mod is not None
                        and getattr(_m1_mod, 'committed_mask', None) is not None
                        and _m1_mod.committed_mask.float().mean().item() > 0.95):
                    del self.shadow[name]
                    self._m1_evicted.add(name)
                    print(f"[M1-EMA-EVICT] {name}: 95%+ committed, shadow freed")
                    continue

            self.shadow[name].mul_(self.decay).add_(
                param.data, alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def apply(self, model: nn.Module):
        backup = {}
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict):
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


class CommitmentMonitor:
    """Tracks ternary weight commitment dynamics and modulates training (R2-A4-M3+M8).

    Monitors:
    - H_T: Global ternary entropy (decreases as weights commit)
    - R_commit: Fraction of weights far from thresholds (increases as weights commit)
    - dH_T/dt: Rate of entropy change (plateau when near zero)

    Actions:
    - LR boost when plateau + high entropy (weights stuck, need stronger signal)
    - LR decay when plateau + low entropy (nearly committed, fine-tune)
    - Early stop when R_commit > threshold and entropy plateauing
    """

    def __init__(
        self,
        check_interval: int = 50,
        plateau_epsilon: float = 0.005,
        plateau_patience: int = 3,
        lr_boost: float = 1.5,
        lr_decay: float = 0.5,
        early_stop_threshold: float = 0.98,
        early_stop: bool = True,
    ):
        self.check_interval = check_interval
        self.plateau_epsilon = plateau_epsilon
        self.plateau_patience = plateau_patience
        self.lr_boost = lr_boost
        self.lr_decay = lr_decay
        self.early_stop_threshold = early_stop_threshold
        self.early_stop = early_stop

        self.entropy_history = []
        self.commit_history = []
        self.dH_dt_history = []

        self.consecutive_plateau_checks = 0
        self.plateau_active = False
        self.current_lr_multiplier = 1.0

        self.should_stop = False
        self.stop_reason = ""

        self.entropy_midpoint = 0.5

    def check(self, model, global_step: int) -> dict:
        """Run commitment check. Call every check_interval steps.

        Returns dict with metrics and any actions taken.
        """
        from ..components.ternary import compute_weight_entropy, compute_commitment_rate

        entropy_stats = compute_weight_entropy(model)
        commitment_stats = compute_commitment_rate(model)

        if not entropy_stats or not commitment_stats:
            return {"action": "skip", "reason": "no ternary layers found"}

        avg_entropy = sum(s["entropy"] for s in entropy_stats) / len(entropy_stats)
        avg_util = sum(s["utilization"] for s in entropy_stats) / len(entropy_stats)
        min_util = min(s["utilization"] for s in entropy_stats)
        avg_committed = sum(s["committed_frac"] for s in commitment_stats) / len(
            commitment_stats
        )

        self.entropy_history.append((global_step, avg_entropy))
        self.commit_history.append((global_step, avg_committed))

        dH_dt = 0.0
        if len(self.entropy_history) >= 2:
            prev_step, prev_H = self.entropy_history[-2]
            step_diff = global_step - prev_step
            if step_diff > 0:
                dH_dt = (avg_entropy - prev_H) / step_diff
        self.dH_dt_history.append((global_step, dH_dt))

        is_plateau = abs(dH_dt) < self.plateau_epsilon
        if is_plateau:
            self.consecutive_plateau_checks += 1
        else:
            self.consecutive_plateau_checks = 0

        self.plateau_active = self.consecutive_plateau_checks >= self.plateau_patience

        action = "none"
        self.current_lr_multiplier = 1.0

        if self.plateau_active:
            if avg_util > self.entropy_midpoint:
                self.current_lr_multiplier = self.lr_boost
                action = "lr_boost"
            else:
                self.current_lr_multiplier = self.lr_decay
                action = "lr_decay"

            if self.early_stop and avg_committed > self.early_stop_threshold:
                self.should_stop = True
                self.stop_reason = (
                    f"Commitment plateau: R_commit={avg_committed:.4f} "
                    f"> {self.early_stop_threshold}, "
                    f"|dH/dt|={abs(dH_dt):.6f} < {self.plateau_epsilon}"
                )
                action = "early_stop"

        result = {
            "step": global_step,
            "avg_entropy": avg_entropy,
            "avg_utilization": avg_util,
            "min_utilization": min_util,
            "avg_committed": avg_committed,
            "dH_dt": dH_dt,
            "is_plateau": is_plateau,
            "plateau_active": self.plateau_active,
            "consecutive_plateau": self.consecutive_plateau_checks,
            "lr_multiplier": self.current_lr_multiplier,
            "action": action,
            "should_stop": self.should_stop,
        }

        return result

    def get_lr_multiplier(self) -> float:
        """Return current LR multiplier based on commitment state."""
        return self.current_lr_multiplier

    def get_summary(self) -> dict:
        """Return summary of commitment dynamics for results dict."""
        if not self.entropy_history:
            return {"commitment_checks": 0}

        return {
            "commitment_checks": len(self.entropy_history),
            "initial_entropy": self.entropy_history[0][1]
            if self.entropy_history
            else 0,
            "final_entropy": self.entropy_history[-1][1] if self.entropy_history else 0,
            "initial_committed": self.commit_history[0][1]
            if self.commit_history
            else 0,
            "final_committed": self.commit_history[-1][1] if self.commit_history else 0,
            "plateau_count": sum(
                1 for _, dh in self.dH_dt_history if abs(dh) < self.plateau_epsilon
            ),
            "early_stopped": self.should_stop,
            "stop_reason": self.stop_reason,
        }


class AdaptiveGradientClipper:
    """Adaptive gradient clipping based on gradient norm statistics (R4-A3-F6).

    tau_t = mu_t + k*sigma_t (clips gradients beyond k standard deviations from mean)
    """

    def __init__(self, k: float = 2.0, beta: float = 0.99):
        self.k = k  # Number of std devs for threshold
        self.beta = beta  # EMA decay rate
        self.grad_norm_mean = None  # EMA of gradient norm
        self.grad_norm_m2 = None  # EMA of squared deviations (for Welford's algorithm)
        self.n = 0  # Step counter

    def clip(self, parameters) -> dict:
        """Compute gradient norm, update statistics, and clip if needed.

        Returns: dict with keys 'grad_norm', 'threshold', 'clipped' (bool)
        """
        # Compute total gradient norm (same as torch.nn.utils.clip_grad_norm_)
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        # Update EMA statistics (Welford's online algorithm for variance)
        if self.grad_norm_mean is None:
            self.grad_norm_mean = total_norm
            self.grad_norm_m2 = 0.0
        else:
            delta = total_norm - self.grad_norm_mean
            self.grad_norm_mean = (
                self.beta * self.grad_norm_mean + (1 - self.beta) * total_norm
            )
            self.grad_norm_m2 = self.beta * self.grad_norm_m2 + (1 - self.beta) * (
                delta**2
            )

        self.n += 1

        # Compute adaptive threshold
        grad_norm_std = (self.grad_norm_m2**0.5) if self.n > 1 else 0.0
        threshold = self.grad_norm_mean + self.k * grad_norm_std

        # Clip if gradient norm exceeds threshold
        clipped = total_norm > threshold
        if clipped:
            clip_coef = threshold / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        return {
            "grad_norm": total_norm,
            "threshold": threshold,
            "clipped": clipped,
            "mean": self.grad_norm_mean,
            "std": grad_norm_std,
        }


class TernaryCommitmentMonitor:
    """Track ternary weight commitment (R4-A4-M3+M8).

    Commitment = fraction of weights that have NOT flipped in last W steps.
    Enables adaptive techniques: curriculum, TBPTT schedule, Fisher sampling.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.prev_ternary = {}  # Previous ternary values per parameter
        self.flip_history = {}  # Circular buffer of flip rates
        self.step_count = 0

    def update(self, model) -> dict:
        """Compute flip rate and commitment for all ternary parameters.

        Returns: dict with keys 'overall_commitment', 'per_layer_commitment', 'flip_rate'
        """
        from ..components.ternary import TernaryLinear

        if not getattr(TernaryCommitmentMonitor, '_m3_logged', False):
            print("[M3-INT8] prev_ternary stored as int8 (75% memory reduction)")
            TernaryCommitmentMonitor._m3_logged = True

        total_flips = 0
        total_weights = 0
        per_layer_stats = {}

        for name, module in model.named_modules():
            if isinstance(module, TernaryLinear):
                # Compute current ternary values
                with torch.no_grad():
                    current_ternary = module.quantize(
                        module.weight
                    ).sign()  # {-1, 0, +1}

                # Compare with previous
                if name in self.prev_ternary:
                    flipped = current_ternary.to(torch.int8) != self.prev_ternary[name]
                    flip_count = flipped.sum().item()
                    weight_count = current_ternary.numel()

                    total_flips += flip_count
                    total_weights += weight_count

                    layer_flip_rate = flip_count / weight_count
                    per_layer_stats[name] = {
                        "flip_rate": layer_flip_rate,
                        "flipped_count": flip_count,
                        "total_count": weight_count,
                    }

                # Store current for next step (M3: INT8 storage, 75% memory reduction)
                self.prev_ternary[name] = current_ternary.to(torch.int8)

        # Overall flip rate for this step
        flip_rate = total_flips / total_weights if total_weights > 0 else 0.0

        # Update flip history (circular buffer)
        if "overall" not in self.flip_history:
            self.flip_history["overall"] = []
        self.flip_history["overall"].append(flip_rate)
        if len(self.flip_history["overall"]) > self.window:
            self.flip_history["overall"].pop(0)

        # Commitment = 1 - average flip rate over window
        avg_flip_rate = sum(self.flip_history["overall"]) / len(
            self.flip_history["overall"]
        )
        commitment = 1.0 - avg_flip_rate

        self.step_count += 1

        return {
            "commitment": commitment,
            "flip_rate": flip_rate,
            "avg_flip_rate": avg_flip_rate,
            "per_layer": per_layer_stats,
            "step": self.step_count,
        }
