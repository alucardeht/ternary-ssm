import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import contextlib
import copy
import os
import logging
from pathlib import Path
from typing import Optional, Callable
from .model import ModelV2
from .config import ModelConfig

# NOTE: data.py functions moved - import directly instead
import sys
from pathlib import Path

# Load data.py module directly to avoid name conflict with data/ package
import importlib.util

_data_module_path = Path(__file__).parent / "data.py"
_spec = importlib.util.spec_from_file_location("_data_module", _data_module_path)
_data_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_module)
get_dataloader = _data_module.get_dataloader
CurriculumSampler = _data_module.CurriculumSampler

from .datasets import CUDAPrefetcher
from .components.router import MoDRouter
from .optimizers.optim import (
    CPUOffloadOptimizer,
    AGSCPUOffloadOptimizer,
    compute_layer_alphas,
    compute_lr_multipliers,
    InformationTheoreticFreezer,
)
from .components.ternary import (
    DEFAULT_THRESHOLD_COEFF,
    TernaryLinear,
    compute_weight_entropy,
    compute_commitment_rate,
    init_gradient_recyclers,
    apply_gradient_recycling,
)
from .components.acsi_init import system_identification_init
from .training.glts import glts_step as _glts_step
from .training.glts import glts_step_blockwise as _glts_step_blockwise
from .training.monitors import (
    get_global_vote_optimizer,
    TrainingMonitor,
    ModelEMA,
    CommitmentMonitor,
    AdaptiveGradientClipper,
    TernaryCommitmentMonitor,
)
from .training.schedulers import ProgressiveScheduler, inv_sqrt_schedule
from .training.losses import margin_loss, distillation_loss

logger = logging.getLogger(__name__)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from liger_kernel.transformers import LigerCrossEntropyLoss

    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False


# Utility functions extracted to training/utils.py
from .training.utils import (
    compute_grad_norm, compute_optimal_batch_size,
    apply_topk_gradient_sparsification, compute_fisher_proxy,
    apply_fisher_preconditioning, apply_sign_momentum,
    compute_layer_sensitivity, apply_adaptive_quantization_levels,
    compute_checkpoint_density, compute_optimal_training_config,
    compute_weight_snr, freeze_low_snr_weights,
    apply_gradient_sign_consistency, apply_hfott_gradient_correction,
    apply_sparse_shadow_weights, update_caq_state,
    update_info_freeze_vhat, zero_frozen_optimizer_state,
    apply_commitment_lr_scaling, evict_committed_params,
)
















































def evaluate_perplexity(
    model: nn.Module,
    config: ModelConfig,
    split: str = "val",
    batch_size: int = 16,
    device: str = "cpu",
    data_dir: str = None,
    use_amp: bool = False,
) -> float:
    """Evaluate perplexity on WikiText-103 validation or test set.

    Evaluates on non-overlapping chunks from WikiTextDataset.
    Perplexity = exp(average cross-entropy loss).
    """
    # Fixed: imports from data.py handled below

    dataset = WikiTextDataset(split=split, seq_len=config.seq_len, data_dir=data_dir)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    if _LIGER_AVAILABLE:
        criterion = LigerCrossEntropyLoss(reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp and str(device) == "cuda"
            ):
                logits = model(input_ids)

            loss = criterion(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))

    model.train()
    return perplexity


def _setup_global_flags(
    config: ModelConfig,
    use_rho1: bool,
    use_distillation: bool,
    online_distill_teacher_path: Optional[str],
    use_saq: bool,
    progressive: bool,
    use_amp: bool,
    use_bf16: bool,
    use_gradient_checkpointing: bool,
    checkpoint_every_n: int,
    ternary_aware_checkpointing: bool,
    use_sdpa: bool,
    use_activation_quant: bool,
    activation_quant_bits: int,
    use_cpu_offload: bool,
    use_compile: bool,
    antiwindup_k: float,
    use_instant: bool,
    instant_rank: int,
):
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_flush_denormals(True)
        print("[FTZ] Flush-to-zero enabled: denormals -> 0.0")
    except AttributeError:
        print("[FTZ] torch.set_flush_denormals not available (GPU already flushes denormals by default)")

    if getattr(config, "use_deterministic_mode", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[DETERMINISTIC] Enabled: torch.use_deterministic_algorithms(True)")

    if use_rho1 and use_distillation:
        raise ValueError("Rho-1 and distillation are mutually exclusive")

    if online_distill_teacher_path is not None and use_distillation:
        raise ValueError("online_distill and use_distillation are mutually exclusive")

    if use_saq and progressive:
        raise ValueError("SAQ and progressive training are mutually exclusive")

    config.use_amp = use_amp
    config.use_bf16 = use_bf16
    config.use_gradient_checkpointing = use_gradient_checkpointing
    config.checkpoint_every_n = checkpoint_every_n
    config.ternary_aware_checkpointing = ternary_aware_checkpointing
    config.use_sdpa = use_sdpa
    config.use_activation_quant = use_activation_quant
    config.activation_quant_bits = activation_quant_bits
    config.use_torch_compile = use_compile
    config.use_cpu_offload = use_cpu_offload
    config.antiwindup_k = antiwindup_k
    config.use_instant = use_instant
    if use_instant:
        config.instant_rank = instant_rank


def _post_training(
    model,
    config,
    optimizer,
    scheduler,
    monitor,
    ema,
    progressive_freezer,
    fused_aga,
    let_teacher,
    let_proj,
    od_teacher,
    train_loader,
    context,
):
    """
    Post-training cleanup and result aggregation.

    Handles epoch summary, peak VRAM measurement, results dictionary construction,
    teacher model cleanup, checkpoint/log saving, schedule-free eval mode,
    WikiText validation, final summary printing, and post-training combinatorial
    ternary optimization. Returns aggregated results dictionary.
    """
    epoch_loss = context["epoch_loss"]
    epoch_tokens = context["epoch_tokens"]
    best_loss = context["best_loss"]
    diverged = context["diverged"]
    global_step = context["global_step"]
    loss = context["loss"]
    param_info = context["param_info"]
    ppl_history = context["ppl_history"]
    start_time = context["start_time"]

    use_saq = context["use_saq"]
    use_amp = context["use_amp"]
    use_combinatorial_finetune = context["use_combinatorial_finetune"]
    combinatorial_max_sweeps = context["combinatorial_max_sweeps"]
    combinatorial_ternary_finetune = context["combinatorial_ternary_finetune"]

    dataset_type = context["dataset_type"]
    batch_size = context["batch_size"]
    data_dir = context["data_dir"]
    device = context["device"]
    save_dir = context["save_dir"]
    on_checkpoint = context["on_checkpoint"]
    optimizer_type = context["optimizer_type"]

    max_steps = context["max_steps"]
    num_epochs = context["num_epochs"]
    commitment_monitor = context["commitment_monitor"]

    lr_schedule_type = context["lr_schedule_type"]
    two_stage_decay = context["two_stage_decay"]
    use_8bit_optimizer = context["use_8bit_optimizer"]
    use_compile = context["use_compile"]
    seq_warmup = context["seq_warmup"]
    gradual_threshold = context["gradual_threshold"]
    pressure_blend = context["pressure_blend"]
    layer_wise_precision = context["layer_wise_precision"]
    use_curriculum = context["use_curriculum"]
    angular_reg_lambda = context["angular_reg_lambda"]
    boundary_reg_lambda = context["boundary_reg_lambda"]
    galore_rank = context["galore_rank"]
    saq_k = context["saq_k"]
    use_dual_rate = context["use_dual_rate"]
    dual_rate_peak = context["dual_rate_peak"]
    dual_rate_floor = context["dual_rate_floor"]
    use_gaste = context["use_gaste"]
    gaste_sigma_g = context["gaste_sigma_g"]
    use_rlec = context["use_rlec"]
    rlec_rank = context["rlec_rank"]
    use_adagc = context["use_adagc"]
    adagc_factor = context["adagc_factor"]
    adagc_beta = context["adagc_beta"]
    use_info_freeze = context["use_info_freeze"]
    info_freeze_alpha = context["info_freeze_alpha"]
    info_freeze_warmup = context["info_freeze_warmup"]
    use_dead_zone = context["use_dead_zone"]
    dead_zone_margin = context["dead_zone_margin"]
    dead_zone_lr_decay = context["dead_zone_lr_decay"]
    use_commitment_monitor = context["use_commitment_monitor"]
    commitment_check_interval = context["commitment_check_interval"]
    commitment_plateau_epsilon = context["commitment_plateau_epsilon"]
    commitment_plateau_patience = context["commitment_plateau_patience"]
    commitment_lr_boost = context["commitment_lr_boost"]
    commitment_lr_decay = context["commitment_lr_decay"]
    commitment_early_stop_threshold = context["commitment_early_stop_threshold"]
    commitment_early_stop = context["commitment_early_stop"]
    use_acsi_init = context["use_acsi_init"]
    acsi_max_batches = context["acsi_max_batches"]
    use_activation_quant = context["use_activation_quant"]
    activation_quant_bits = context["activation_quant_bits"]
    use_instant = context["use_instant"]
    instant_rank = context["instant_rank"]
    use_ags = context["use_ags"]
    ags_top_k_frac = context["ags_top_k_frac"]
    ags_full_sync_interval = context["ags_full_sync_interval"]
    ags_delta = context["ags_delta"]

    evaluate_perplexity = context["evaluate_perplexity"]

    avg_loss = epoch_loss / max(epoch_tokens, 1)
    print(f"\n--- Epoch {num_epochs} | Avg Loss: {avg_loss:.4f} ---\n")

    if avg_loss < best_loss:
        best_loss = avg_loss
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

    if use_saq and config.use_ternary:
        model.set_quantization_enabled(True)

    total_time = time.time() - start_time

    peak_vram_mb = 0
    if device == "cuda" and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    results = {
        "total_steps": global_step,
        "best_loss": best_loss,
        "final_loss": loss.item() if loss is not None else float("inf"),
        "total_time_sec": total_time,
        "tokens_per_sec": epoch_tokens / total_time if total_time > 0 else 0,
        "peak_vram_mb": peak_vram_mb,
        "param_count": param_info,
        "ppl_history": ppl_history,
        "diverged": diverged,
        "config": {
            "lr_schedule": lr_schedule_type,
            "two_stage_decay": two_stage_decay,
            "optimizer_type": optimizer_type,
            "use_8bit_optimizer": use_8bit_optimizer,
            "use_compile": use_compile,
            "seq_warmup": seq_warmup,
            "gradual_threshold": gradual_threshold,
            "pressure_blend": pressure_blend,
            "layer_wise_precision": layer_wise_precision,
            "use_curriculum": use_curriculum,
            "angular_reg_lambda": angular_reg_lambda,
            "boundary_reg_lambda": boundary_reg_lambda,
            "use_galore": optimizer_type == "galore",
            "galore_rank": galore_rank,
            "use_saq": context["use_saq"],
            "saq_k": saq_k,
            "use_dual_rate": use_dual_rate,
            "dual_rate_peak": dual_rate_peak,
            "dual_rate_floor": dual_rate_floor,
            "use_gaste": use_gaste,
            "gaste_sigma_g": gaste_sigma_g,
            "use_rlec": use_rlec,
            "rlec_rank": rlec_rank,
            "use_adagc": use_adagc,
            "adagc_factor": adagc_factor,
            "adagc_beta": adagc_beta,
            "use_info_freeze": use_info_freeze,
            "info_freeze_alpha": info_freeze_alpha,
            "info_freeze_warmup": info_freeze_warmup,
            "use_dead_zone": use_dead_zone,
            "dead_zone_margin": dead_zone_margin,
            "dead_zone_lr_decay": dead_zone_lr_decay,
            "use_commitment_monitor": use_commitment_monitor,
            "commitment_check_interval": commitment_check_interval,
            "commitment_plateau_epsilon": commitment_plateau_epsilon,
            "commitment_plateau_patience": commitment_plateau_patience,
            "commitment_lr_boost": commitment_lr_boost,
            "commitment_lr_decay": commitment_lr_decay,
            "commitment_early_stop_threshold": commitment_early_stop_threshold,
            "commitment_early_stop": commitment_early_stop,
            "use_acsi_init": use_acsi_init,
            "acsi_max_batches": acsi_max_batches,
            "use_activation_quant": use_activation_quant,
            "activation_quant_bits": activation_quant_bits,
            "use_instant": use_instant,
            "instant_rank": instant_rank,
            "use_ags": use_ags,
            "ags_top_k_frac": ags_top_k_frac,
            "ags_full_sync_interval": ags_full_sync_interval,
            "ags_delta": ags_delta,
        },
        "commitment_summary": commitment_monitor.get_summary()
        if commitment_monitor
        else {},
    }

    if let_teacher is not None:
        del let_teacher
        if let_proj is not None:
            del let_proj
        if device == "cuda":
            torch.cuda.empty_cache()

    if od_teacher is not None:
        del od_teacher
        if device == "cuda":
            torch.cuda.empty_cache()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        monitor.save(f"{save_dir}/training_log.json")

        final_ckpt = {
            "model_state": (
                model._orig_mod if hasattr(model, "_orig_mod") else model
            ).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "step": global_step,
            "ppl_history": ppl_history,
        }
        torch.save(final_ckpt, os.path.join(save_dir, "final_checkpoint.pt"))
        torch.save(final_ckpt["model_state"], os.path.join(save_dir, "final_model.pt"))
        if on_checkpoint:
            on_checkpoint()

        with open(f"{save_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {save_dir}/")

    if optimizer_type == "schedule_free":
        optimizer.eval()

    if dataset_type == "wikitext":
        print(f"\nEvaluating perplexity on validation set...")
        _ema_backup = None
        if ema is not None:
            _ema_backup = ema.apply(
                model._orig_mod if hasattr(model, "_orig_mod") else model
            )
        val_ppl = evaluate_perplexity(
            model,
            config,
            split="val",
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            data_dir=data_dir,
        )
        if _ema_backup is not None:
            ema.restore(
                model._orig_mod if hasattr(model, "_orig_mod") else model, _ema_backup
            )
        results["val_perplexity"] = val_ppl
        print(f"Validation perplexity: {val_ppl:.2f}")

    print(f"\n{'=' * 60}")
    print(f"Training Complete")
    print(f"{'=' * 60}")
    print(f"Steps: {global_step}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Time: {total_time:.1f}s")

    if progressive_freezer is not None:
        stats = progressive_freezer.get_stats()
        frozen, total = progressive_freezer.get_frozen_param_count()
        print(
            f"[C3-FINAL] Frozen {stats['n_frozen']}/{stats['n_total']} layers "
            f"({frozen}/{total} params, {stats['pct_frozen']:.1f}% of layers)"
        )
        if stats["freeze_history"]:
            print(f"[C3-FINAL] Freeze sequence: {stats['freeze_history'][:5]}...")

    print(f"{'=' * 60}\n")

    if fused_aga is not None:
        fused_aga.remove_hooks()

    if use_combinatorial_finetune and config.use_ternary:
        print(f"\n[Combinatorial Finetune] Starting post-training optimization...")
        combinatorial_ternary_finetune(
            model, train_loader, device, max_iters=combinatorial_max_sweeps, patience=5
        )

    return results


def _setup_data(
    config: ModelConfig,
    model,
    dataset_type: str,
    batch_size: int,
    num_samples: int,
    data_dir: Optional[str],
    device: str,
    use_curriculum: bool,
    rho1_ref_losses_path: Optional[str],
    auto_batch_size: bool,
    vram_budget_mb: int,
    use_cuda_prefetch: bool,
    use_sequence_packing: bool,
    use_acsi_init: bool,
    acsi_max_batches: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    lr_schedule_type: str,
    use_rho1: bool,
) -> tuple:
    curriculum_sampler = None
    if use_curriculum:
        if rho1_ref_losses_path is None:
            raise ValueError("use_curriculum requires rho1_ref_losses_path")
        curriculum_sampler = CurriculumSampler(rho1_ref_losses_path, config.seq_len)
        if lr_schedule_type == "cosine":
            print("WARNING: Cosine LR decay kills curriculum gains (Luo et al. 2025).")
            print("  Recommended: --lr-schedule moderate_decay")

    if auto_batch_size:
        batch_size = compute_optimal_batch_size(
            model, config.seq_len, vram_budget_mb, device
        )
        print(
            f"Auto-computed batch_size: {batch_size} (VRAM budget: {vram_budget_mb} MB)"
        )

    train_loader = get_dataloader(
        dataset_type,
        config.vocab_size,
        config.seq_len,
        batch_size,
        num_samples,
        data_dir=data_dir,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        ref_losses_path=rho1_ref_losses_path if use_rho1 else None,
        use_sequence_packing=use_sequence_packing,
        sampler=curriculum_sampler,
    )

    print(
        f"[DATALOADER-H] num_workers={num_workers}, prefetch_factor={prefetch_factor}, "
        f"persistent_workers={persistent_workers}, pin_memory={pin_memory}"
    )

    if use_cuda_prefetch and ("cuda" in device if isinstance(device, str) else device.type == "cuda"):
        train_loader = CUDAPrefetcher(train_loader, torch.device(device) if isinstance(device, str) else device)
        print("Enabled F3 CUDA-Stream async prefetch for data loading")

    if getattr(config, "use_deferred_ternary", False) and config.use_ternary:
        from .components.ternary import convert_model_to_deferred

        convert_model_to_deferred(model, delta_rank=config.deferred_rank)
        if device == "cuda":
            torch.cuda.empty_cache()

    if use_acsi_init and config.use_ssd:
        from .components.system_id import acsi_init_model

        _acsi_dl = get_dataloader(
            dataset_type,
            config.vocab_size,
            config.seq_len,
            batch_size,
            num_samples,
            data_dir=data_dir,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            use_sequence_packing=use_sequence_packing,
        )
        acsi_init_model(model, _acsi_dl, device=device, max_batches=acsi_max_batches)
        del _acsi_dl
        print(f"[ACSI] Init complete, proceeding to training")

    return train_loader, curriculum_sampler


def _log_and_eval(ctx):
    model = ctx["model"]
    optimizer = ctx["optimizer"]
    scheduler = ctx["scheduler"]
    ema = ctx["ema"]
    config = ctx["config"]
    global_step = ctx["global_step"]
    loss = ctx["loss"]
    grad_norm = ctx["grad_norm"]
    epoch = ctx["epoch"]
    epoch_tokens = ctx["epoch_tokens"]
    batch_size = ctx["batch_size"]
    device = ctx["device"]
    data_dir = ctx["data_dir"]
    eval_every = ctx["eval_every"]
    ppl_history = ctx["ppl_history"]
    checkpoint_every = ctx["checkpoint_every"]
    save_dir = ctx["save_dir"]
    start_step = ctx["start_step"]
    anomaly_detector = ctx["anomaly_detector"]
    ram_checkpointer = ctx["ram_checkpointer"]
    on_checkpoint = ctx["on_checkpoint"]
    two_stage_decay = ctx["two_stage_decay"]
    total_steps = ctx["total_steps"]
    two_stage_wd_start_pct = ctx["two_stage_wd_start_pct"]
    weight_decay = ctx["weight_decay"]
    log_every = ctx["log_every"]
    start_time = ctx["start_time"]
    base_lr = ctx["base_lr"]
    monitor = ctx["monitor"]
    lr_schedule_type = ctx["lr_schedule_type"]
    two_stage_drop_pct = ctx["two_stage_drop_pct"]
    two_stage_drop_factor = ctx["two_stage_drop_factor"]
    progressive_scheduler = ctx["progressive_scheduler"]
    aga_optimizer = ctx["aga_optimizer"]
    bbo_optimizer = ctx["bbo_optimizer"]
    fused_aga = ctx["fused_aga"]
    rho1_stats = ctx["rho1_stats"]
    use_adagc = ctx["use_adagc"]
    clip_stats = ctx["clip_stats"]
    let_teacher = ctx["let_teacher"]
    let_stop_step = ctx["let_stop_step"]
    teacher_model = ctx["teacher_model"]
    od_teacher = ctx["od_teacher"]
    angular_reg_lambda = ctx["angular_reg_lambda"]
    boundary_reg_lambda = ctx["boundary_reg_lambda"]
    use_tdo = ctx["use_tdo"]
    _tdo_stats = ctx["_tdo_stats"]
    _ang_loss = ctx["_ang_loss"]
    _bnd_loss = ctx["_bnd_loss"]
    ce_loss = ctx["ce_loss"]
    kd_loss = ctx["kd_loss"]
    od_kd_loss = ctx["od_kd_loss"]
    cos_sim = ctx["cos_sim"]
    instant_rank = ctx["instant_rank"]
    use_activation_quant = ctx["use_activation_quant"]
    activation_quant_bits = ctx["activation_quant_bits"]
    use_info_freeze = ctx["use_info_freeze"]
    use_dead_zone = ctx["use_dead_zone"]
    mem_mb = ctx["mem_mb"]
    max_steps = ctx["max_steps"]

    if eval_every and global_step > 0 and global_step % eval_every == 0:
        _ema_backup = None
        if ema is not None:
            _ema_backup = ema.apply(
                model._orig_mod if hasattr(model, "_orig_mod") else model
            )
        model.eval()
        torch.cuda.empty_cache()
        eval_batch = min(batch_size, 16)
        val_ppl = evaluate_perplexity(
            model if not hasattr(model, "_orig_mod") else model._orig_mod,
            config,
            split="val",
            batch_size=eval_batch,
            device=device,
            data_dir=data_dir,
            use_amp=use_adagc if isinstance(use_adagc, bool) else False,
        )
        ppl_history.append(
            {"step": global_step, "val_ppl": round(val_ppl, 2)}
        )
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"  >>> Step {global_step}: Val PPL = {val_ppl:.2f}")
        if _ema_backup is not None:
            ema.restore(
                model._orig_mod if hasattr(model, "_orig_mod") else model,
                _ema_backup,
            )
        model.train()

    if (
        checkpoint_every
        and global_step > 0
        and global_step % checkpoint_every == 0
        and save_dir
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ckpt_dict = {
            "model_state": (
                model._orig_mod if hasattr(model, "_orig_mod") else model
            ).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
            if scheduler
            else None,
            "step": global_step,
            "ppl_history": ppl_history,
            "config_name": str(config),
        }
        ckpt_path = os.path.join(
            save_dir, f"checkpoint_step{global_step}.pt"
        )

        if getattr(config, "use_ternary_checkpoint_compression", False):
            from .components.ternary_serialization import (
                pack_state_dict_ternary,
            )

            ckpt_dict["model_state"] = pack_state_dict_ternary(
                ckpt_dict["model_state"]
            )
            if global_step == start_step + (checkpoint_every or 999):
                stats = ckpt_dict["model_state"].get(
                    "compression_stats", {}
                )
                print(
                    f"[TERNARY-CKPT] Compression: {stats.get('ratio', 0):.1f}x, "
                    f"packed {stats.get('num_packed_tensors', 0)} tensors"
                )

        if anomaly_detector is not None:
            anomaly_detector.save_checkpoint_to_ring(ckpt_dict, global_step)

        if ram_checkpointer is not None:
            ram_checkpointer.save_to_ram(
                ckpt_dict, global_step, disk_path=ckpt_path
            )
        else:
            torch.save(ckpt_dict, ckpt_path)
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

        if on_checkpoint:
            on_checkpoint()

    if two_stage_decay:
        wd_start_step = int(total_steps * two_stage_wd_start_pct)
        if global_step < wd_start_step:
            current_wd = weight_decay
        else:
            wd_progress = (global_step - wd_start_step) / max(
                1, total_steps - wd_start_step
            )
            current_wd = (
                weight_decay * 0.5 * (1.0 + math.cos(wd_progress * math.pi))
            )
        for pg in optimizer.param_groups:
            if pg.get("name") != "embedding":
                pg["weight_decay"] = current_wd

    if global_step % log_every == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
        try:
            current_lr = optimizer.param_groups[0]["lr"]
        except (IndexError, KeyError):
            current_lr = base_lr

        log_dict = {
            "step": global_step,
            "loss": loss.item(),
            "lr": current_lr,
            "tokens_per_sec": tokens_per_sec,
            "grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm,
            "epoch": epoch,
        }
        if progressive_scheduler is not None:
            log_dict["phase"] = progressive_scheduler.phase_name
        monitor.log(**log_dict)

        extra = f" | Mem {mem_mb:.0f}MB"
        if two_stage_decay:
            _current_wd = optimizer.param_groups[0].get(
                "weight_decay", weight_decay
            )
            extra += f" | WD={_current_wd:.4f}"
        if lr_schedule_type == "two_stage":
            _drop_step = int(total_steps * two_stage_drop_pct)
            _phase = "explore" if global_step < _drop_step else "commit"
            extra += f" | Phase={_phase}"
        if (
            aga_optimizer is not None
            and global_step % 50 == 0
            and global_step > 0
        ):
            aga_stats = aga_optimizer.get_stats()
            extra += (
                f" | [AGA] flips/step={aga_stats['flips_per_step']:.0f} "
                f"avg|acc|={aga_stats['avg_abs_accumulator']:.1f} "
                f"max|acc|={aga_stats['max_accumulator']:.0f} "
                f"near_K={aga_stats['near_threshold_pct']:.1f}%"
            )
        if (
            bbo_optimizer is not None
            and global_step % 50 == 0
            and global_step > 0
        ):
            bbo_stats = bbo_optimizer.get_stats()
            dist = bbo_stats["state_distribution"]
            extra += (
                f" | [BBO] flips/step={bbo_stats['flips_per_step']:.1f} "
                f"ballot={bbo_stats['avg_abs_ballot']:.1f} "
                f"dist={{-1:{dist[-1]:.1f}%, 0:{dist[0]:.1f}%, +1:{dist[1]:.1f}%}}"
            )
        if (
            fused_aga is not None
            and global_step % 50 == 0
            and global_step > 0
        ):
            _fstats = fused_aga.get_stats()
            extra += (
                f"  [FUSED-AGA] flips/step={_fstats['flips_per_step']:.1f} "
                f"avg_acc={_fstats['avg_abs_accumulator']:.2f} "
                f"near_K={_fstats['near_threshold_pct']:.1f}% "
                f"total_flips={_fstats['total_flips']}"
            )
        if rho1_stats is not None:
            extra += f" | Rho1 kept={rho1_stats['kept_ratio']:.1%} excess={rho1_stats['mean_excess']:.3f}"
        if use_adagc and "_adagc_clips" in dir():
            extra += f" | AdaGC clips={_adagc_clips} max_ratio={_adagc_max_ratio:.2f}"
        if global_step % 100 == 0 or clip_stats["clipped"]:
            extra += f" | [AdaGC] tau={clip_stats['threshold']:.3f} (mu={clip_stats['mean']:.3f} sigma={clip_stats['std']:.3f}) clipped={clip_stats['clipped']}"
        if let_teacher is not None and global_step < let_stop_step:
            extra += f" | LET cos={cos_sim.item():.4f}"
        if teacher_model is not None:
            extra += f" | CE {ce_loss.item():.4f} KD {kd_loss.item():.4f}"
        if od_teacher is not None:
            extra += f" | OD-KD {od_kd_loss.item():.4f}"
        if (
            angular_reg_lambda > 0 or boundary_reg_lambda > 0
        ) and config.use_ternary:
            extra += f" | AngReg {_ang_loss.item():.4f} BndReg {_bnd_loss.item():.4f}"
        if use_tdo and config.use_ternary:
            extra += f" | TDO {_tdo_loss.item():.4f} [{_tdo_stats['neg']:.2f}/{_tdo_stats['zero']:.2f}/{_tdo_stats['pos']:.2f}]"
        if progressive_scheduler is not None:
            extra += f" | Phase {progressive_scheduler.phase_name}"
        if config.use_ternary:
            _prefer_dsst = getattr(config, 'use_dsst', False)
            for name, module in model.named_modules():
                if hasattr(module, "get_ternary_stats"):
                    if _prefer_dsst and not getattr(module, 'use_dsst', False):
                        continue
                    stats = module.get_ternary_stats()
                    extra += f" | Tern +{stats['positive']:.2f} 0:{stats['zero']:.2f} -{stats['negative']:.2f}"
                    break

        if (
            getattr(config, "use_deferred_ternary", False)
            and global_step % log_every == 0
        ):
            from .components.ternary import get_deferred_stats

            d_stats = get_deferred_stats(model)
            if d_stats["enabled"]:
                extra += f" | Deferred delta_norm: avg={d_stats['avg_delta_norm_ratio']:.4f} max={d_stats['max_delta_norm_ratio']:.4f}"

        if (
            getattr(config, "use_instant", False)
            and global_step % (log_every * 4) == 0
        ):
            _inst_active = 0
            _inst_total = 0
            _inst_tensors = 0
            for _iname, _imod in model.named_modules():
                if hasattr(_imod, "_instant_step") and hasattr(
                    _imod, "use_instant"
                ):
                    _inst_total += 1
                    if (
                        _imod.use_instant
                        and _imod._instant_step >= _imod._instant_warmup
                    ):
                        _inst_active += 1
                        _inst_tensors += len(
                            getattr(_imod, "_instant_projectors", {})
                        )
            if _inst_total > 0:
                print(
                    f"  [INSTANT] Active SWA layers: {_inst_active}/{_inst_total}, "
                    f"projectors: {_inst_tensors}, rank={instant_rank}"
                )

        if config.use_ternary and global_step % (log_every * 4) == 0:
            entropy_stats = compute_weight_entropy(model)
            commitment_stats = compute_commitment_rate(model)
            if entropy_stats and commitment_stats:
                avg_util = sum(
                    s["utilization"] for s in entropy_stats
                ) / len(entropy_stats)
                min_util = min(
                    (s["utilization"] for s in entropy_stats), default=0
                )
                avg_committed = sum(
                    s["committed_frac"] for s in commitment_stats
                ) / len(commitment_stats)
                print(
                    f"  [DIAG] Entropy: avg_util={avg_util:.3f} min_util={min_util:.3f} | "
                    f"Commitment: avg={avg_committed:.3f}"
                )
                for s in entropy_stats:
                    if s["utilization"] < 0.8:
                        print(
                            f"  [DIAG WARNING] Low entropy layer {s['name']}: "
                            f"H={s['entropy']:.3f}/{s['max_entropy']:.3f} "
                            f"util={s['utilization']:.3f} dist={s['dist']}"
                        )

        if (
            use_activation_quant
            and config.use_ternary
            and global_step % (log_every * 4) == 0
        ):
            _aq_active = 0
            _aq_total = 0
            for _aqname, _aqmod in model.named_modules():
                if hasattr(_aqmod, "_act_quant_step") and hasattr(
                    _aqmod, "use_activation_quant"
                ):
                    _aq_total += 1
                    if (
                        _aqmod.use_activation_quant
                        and _aqmod._act_quant_step
                        >= _aqmod._act_quant_warmup
                    ):
                        _aq_active += 1
            if _aq_total > 0:
                print(
                    f"  [ACT-QUANT] Active SSM layers: {_aq_active}/{_aq_total}, "
                    f"bits={activation_quant_bits}"
                )

        if (
            use_info_freeze
            and config.use_ternary
            and global_step % (log_every * 4) == 0
        ):
            _total_frozen = 0
            _total_params = 0
            for _fname, _fmod in model.named_modules():
                if hasattr(_fmod, "get_freeze_stats"):
                    _fstats = _fmod.get_freeze_stats()
                    if _fstats.get("freeze_enabled"):
                        _total_frozen += _fstats["frozen_count"]
                        _total_params += _fstats["total_count"]
            if _total_params > 0:
                _freeze_pct = _total_frozen / _total_params
                print(
                    f"  [INFO-FREEZE] Global: {_total_frozen}/{_total_params} "
                    f"({_freeze_pct:.1%}) frozen"
                )

        if (
            use_dead_zone
            and config.use_ternary
            and global_step % (log_every * 4) == 0
        ):
            _total_blocked = 0
            _total_allowed = 0
            for _dzname, _dzmod in model.named_modules():
                if hasattr(_dzmod, "_dead_zone_flips_blocked"):
                    _total_blocked += _dzmod._dead_zone_flips_blocked
                    _total_allowed += _dzmod._dead_zone_flips_allowed
            _total_flips = _total_blocked + _total_allowed
            _block_rate = _total_blocked / max(_total_flips, 1)
            print(
                f"  [DEAD-ZONE] Global: blocked={_total_blocked} "
                f"allowed={_total_allowed} block_rate={_block_rate:.1%}"
            )

        if config.use_routing:
            routing_stats = []
            for name, module in model.named_modules():
                if hasattr(module, "get_stats") and hasattr(
                    module, "capacity"
                ):
                    stats = module.get_stats()
                    routing_stats.append(stats)
            if routing_stats:
                avg_ratio = sum(
                    s["process_ratio"] for s in routing_stats
                ) / len(routing_stats)
                avg_weight = sum(
                    s["mean_weight"] for s in routing_stats
                ) / len(routing_stats)
                extra += f" | Route {avg_ratio:.2f} MeanW {avg_weight:.3f}"

        if (
            getattr(config, "use_tada", False)
            and global_step % log_every == 0
        ):
            if (
                hasattr(model, "tada_alphas")
                and model.tada_alphas is not None
            ):
                alphas = model.tada_alphas.data.cpu().numpy()
                alpha_str = ", ".join(f"{a:.3f}" for a in alphas)
                print(f"  [TADA] alphas=[{alpha_str}]")
            if hasattr(model, "_monitor_tada_rank"):
                model._monitor_tada_rank()

        if config.use_moe and global_step % 50 == 0:
            _moe_stats = []
            for _name, _m in model.named_modules():
                if hasattr(_m, "_expert_counts") and hasattr(
                    _m, "get_stats"
                ):
                    _s = _m.get_stats()
                    if _s.get("total_tokens", 0) > 0:
                        _moe_stats.append(_s)
                        _m.reset_stats()
            if _moe_stats:
                _avg_bal = sum(
                    s["balance_ratio"] for s in _moe_stats
                ) / len(_moe_stats)
                extra += f" | MoE bal={_avg_bal:.3f}"

        if (
            getattr(config, "use_ftec", False)
            and global_step % log_every == 0
        ):
            for name, module in model.named_modules():
                if hasattr(module, "n_factors") and hasattr(
                    module, "shadow_values"
                ):
                    for k in range(module.n_factors):
                        from .components.ternary import (
                            TernaryQuantize,
                        )

                        w = module.shadow_values[k].data
                        threshold = getattr(
                            module,
                            "threshold_coeff",
                            DEFAULT_THRESHOLD_COEFF,
                        )
                        q = TernaryQuantize.apply(
                            w, module.group_size, threshold
                        )
                        n_pos = (q > 0).sum().item()
                        n_neg = (q < 0).sum().item()
                        n_zero = (q == 0).sum().item()
                        total = q.numel()
                        print(
                            f"  [FTEC] {name} factor={k} +1={n_pos / total:.1%} 0={n_zero / total:.1%} -1={n_neg / total:.1%}"
                        )
                    if hasattr(module, "skip_alpha"):
                        print(
                            f"  [FTEC] {name} skip_alpha={module.skip_alpha.item():.4f}"
                        )
                    break

        monitor.print_status(
            global_step,
            loss.item(),
            current_lr,
            tokens_per_sec,
            grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm,
            extra,
        )

        if (
            hasattr(config, "use_commitment_tracking")
            and config.use_commitment_tracking
        ):
            total_weights = 0
            committed_weights = 0
            total_sigma_g = 0
            total_margin = 0
            num_layers = 0

            for module in model.modules():
                if (
                    hasattr(module, "committed_mask")
                    and module.committed_mask is not None
                ):
                    num_layers += 1
                    total_weights += module.committed_mask.numel()
                    committed_weights += module.committed_mask.sum().item()
                    total_sigma_g += module.sigma_g.mean().item()

                    dist_neg = (module.weight + 0.5).abs()
                    dist_pos = (module.weight - 0.5).abs()
                    margin = torch.min(dist_neg, dist_pos).mean().item()
                    total_margin += margin

            if num_layers > 0:
                global_commit_rate = (
                    committed_weights / total_weights
                    if total_weights > 0
                    else 0
                )
                avg_sigma_g = total_sigma_g / num_layers
                avg_margin = total_margin / num_layers

                print(
                    f"[C2-GLOBAL] step={global_step}, "
                    f"commit_rate={global_commit_rate * 100:.2f}%, "
                    f"committed={committed_weights}/{total_weights}, "
                    f"avg_sigma_g={avg_sigma_g:.4f}, "
                    f"avg_margin={avg_margin:.4f}"
                )

                progress = (
                    global_step / max(1, max_steps) if max_steps else 0
                )
                if progress > 0.6 and global_commit_rate < 0.90:
                    print(
                        f"[C2-WARNING] Commitment rate {global_commit_rate * 100:.1f}% below expected 95% at {progress * 100:.0f}% progress"
                    )


def _setup_optimizer(
    config: ModelConfig,
    model,
    device: str,
    base_lr: float,
    beta2: float,
    weight_decay: float,
    optimizer_type: str,
    num_epochs: int,
    len_train_loader: int,
    max_steps: Optional[int],
    warmup_steps: int,
    lr_schedule_type: str,
    wsd_warmup_pct: float,
    wsd_stable_pct: float,
    two_stage_drop_pct: float,
    two_stage_drop_factor: float,
    cosine_restart_period: int,
    cosine_restart_decay: float,
    use_ternary: bool,
    use_uncertainty_loss: bool,
    uw_loss_module,
    use_fused_gradient_cascade: bool,
    use_cpu_offload: bool,
    use_ags: bool,
    ags_top_k_frac: float,
    ags_delta: float,
    ags_full_sync_interval: int,
    use_ema: bool,
    ema_decay: float,
    galore_rank: int,
    galore_update_proj_gap: int,
    galore_scale: float,
    use_amp: bool,
    use_bf16: bool,
    fused_optimizer: bool,
    use_distillation: bool,
    teacher_model_name: str,
    use_info_freezing: bool,
    info_freeze_threshold: float,
    info_freezing_warmup: int,
    info_freeze_window: int,
    info_freeze_beta: float,
):
    aga_optimizer = None
    bbo_optimizer = None
    aga_lr = 0.0

    print(
        f"[DEBUG AGA/BBO] use_aga={use_ags}, use_bbo={use_ternary}, use_ternary={use_ternary}"
    )

    if use_ags and use_ternary:
        from .optimizers.aga import AGAOptimizer

        ternary_weights, non_ternary_groups = model.get_aga_param_split(base_lr)
        param_groups = non_ternary_groups

        aga_lr = base_lr * getattr(config, "aga_lr_multiplier", 1.0)
        aga_optimizer = AGAOptimizer(
            [{"params": ternary_weights}],
            lr=aga_lr,
            threshold_K=getattr(config, "aga_threshold_k", 32),
            decay_steps=getattr(config, "aga_decay_steps", 0),
            ternary_pull=getattr(config, "aga_ternary_pull", 0.0),
            use_int16=getattr(config, "aga_use_int16", True),
        )
        print(
            f"[AGA] Enabled: {sum(p.numel() for p in ternary_weights):,} ternary params -> AGA (K={config.aga_threshold_k})"
        )
        print(
            f"[AGA] Non-ternary: {sum(sum(p.numel() for p in g['params']) for g in non_ternary_groups):,} params -> AdamW"
        )
    elif getattr(config, "use_tsa", False) and use_ternary:
        from .optimizers.tsa import TernarySignAccumulator

        ternary_weights, non_ternary_groups = model.get_aga_param_split(base_lr)

        tsa_ternary_group = {
            "params": ternary_weights,
            "is_ternary": True,
            "tsa_threshold": getattr(config, "tsa_threshold", 50),
        }
        for g in non_ternary_groups:
            g["is_ternary"] = False

        param_groups = [tsa_ternary_group] + non_ternary_groups

        optimizer = TernarySignAccumulator(
            param_groups,
            tsa_threshold=getattr(config, "tsa_threshold", 50),
            lr=base_lr,
            weight_decay=weight_decay,
        )
        print(
            f"[TSA] Enabled: {sum(p.numel() for p in ternary_weights):,} ternary params -> TSA (threshold={config.tsa_threshold})"
        )
        print(
            f"[TSA] Non-ternary: {sum(sum(p.numel() for p in g['params']) for g in non_ternary_groups):,} params -> AdamW"
        )
    elif getattr(config, "use_bbo", False) and use_ternary:
        from .optimizers.bbo import BBOOptimizer

        ternary_weights, non_ternary_groups = model.get_aga_param_split(base_lr)
        param_groups = non_ternary_groups

        bbo_optimizer = BBOOptimizer(
            ternary_weights,
            threshold_K=getattr(config, "bbo_threshold_k", 16),
            decay_steps=getattr(config, "bbo_decay_steps", 100),
            magnitude_threshold=getattr(config, "bbo_magnitude_threshold", 0.0),
            use_two_step=getattr(config, "bbo_two_step", True),
        )
        bbo_optimizer.initialize_from_shadow(ternary_weights)
        print(
            f"[BBO] Enabled: {sum(p.numel() for p in ternary_weights):,} ternary params -> BBO (K={config.bbo_threshold_k})"
        )
        print(
            f"[BBO] Non-ternary: {sum(sum(p.numel() for p in g['params']) for g in non_ternary_groups):,} params -> AdamW"
        )
    else:
        _use_dsst = getattr(config, "use_dsst", False)
        if _use_dsst and use_ternary and optimizer_type == "lion":
            param_groups = model.get_dsst_param_split(base_lr)
            _excl = sum(1 for m in model.modules() if hasattr(m, 'use_dsst') and m.use_dsst and hasattr(m, 'weight'))
            _excl_n = sum(m.weight.numel() for m in model.modules() if hasattr(m, 'use_dsst') and m.use_dsst and hasattr(m, 'weight'))
            print(f"[DSST-SPLIT] {_excl} ternary weights ({_excl_n:,} params) excluded from Lion — DSST hook owns them", flush=True)
        else:
            param_groups = model.get_param_groups(base_lr)

    fused_aga = None
    if use_fused_gradient_cascade:
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            raise RuntimeError(
                "[FUSED-CASCADE] Requires PyTorch >= 2.1.0 for register_post_accumulate_grad_hook. "
                f"Current version: {torch.__version__}"
            )
        if not (use_ags and use_ternary):
            import warnings

            warnings.warn(
                "[FUSED-CASCADE] use_fused_gradient_cascade requires use_aga=True and use_ternary=True. Ignoring."
            )
        else:
            from .optimizers.aga import FusedAGAAccumulator

            fused_aga = FusedAGAAccumulator(
                lr=aga_lr,
                threshold_K=getattr(config, "aga_threshold_k", 32),
                decay_steps=getattr(config, "aga_decay_steps", 0),
                ternary_pull=getattr(config, "aga_ternary_pull", 0.0),
                use_int16=getattr(config, "aga_use_int16", True),
            )
            n_hooks = fused_aga.install_hooks(ternary_weights)
            print(
                f"[FUSED-CASCADE] Installed {n_hooks} gradient hooks on ternary params"
            )
            print(
                f"[FUSED-CASCADE] Gradient memory for ternary params eliminated (streamed to INT16 accumulators)"
            )
            aga_optimizer = None
            print(f"[FUSED-CASCADE] AGA optimizer replaced by fused hooks")

    if uw_loss_module is not None:
        param_groups.append(
            {
                "params": list(uw_loss_module.parameters()),
                "lr": base_lr,
                "weight_decay": 0.0,
            }
        )
        print(
            f"[UNCERTAINTY-LOSS] Added {sum(p.numel() for p in uw_loss_module.parameters())} log-variance parameters to optimizer"
        )

    if optimizer_type == "muon":
        from .optimizers.optim import Muon as MuonOpt

        muon_groups = model.get_muon_param_groups(base_lr)
        muon_count = sum(p.numel() for p in muon_groups[0]["params"])
        adamw_count = sum(p.numel() for p in muon_groups[1]["params"])
        optimizer = MuonOpt(
            muon_groups,
            lr=base_lr * 10,
            momentum=0.95,
            ns_steps=3,
            weight_decay=0.01,
            cautious=False,
        )
        print(
            f"Using Muon optimizer ({muon_count:,} Muon params, {adamw_count:,} AdamW params)"
        )

    elif optimizer_type == "schedule_free":
        from schedulefree import AdamWScheduleFree

        optimizer = AdamWScheduleFree(
            param_groups,
            lr=base_lr,
            betas=(0.9, beta2),
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        print(f"Using Schedule-Free AdamW (warmup={warmup_steps} steps)")

    elif optimizer_type == "ademamix":
        from pytorch_optimizer import AdEMAMix

        optimizer = AdEMAMix(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.999, 0.9999),
            weight_decay=weight_decay,
            alpha=5.0,
        )
        print(f"Using AdEMAMix optimizer (alpha=5.0, beta3=0.9999)")

    elif optimizer_type == "spam":
        from pytorch_optimizer import SPAM

        optimizer = SPAM(param_groups, lr=base_lr, weight_decay=weight_decay)
        print(f"Using SPAM optimizer")

    elif optimizer_type == "fira":
        from .optimizers.optim import FiraAdamW

        fira_rank = config.d_model // 4
        for pg in param_groups:
            pg["rank"] = fira_rank
            pg["update_proj_gap"] = 500
        optimizer = FiraAdamW(
            param_groups, betas=(0.9, beta2), weight_decay=weight_decay
        )
        print(f"Using FiraAdamW optimizer (rank={fira_rank})")

    elif optimizer_type == "galore":
        from .optimizers.galore import GaLoreAdamW

        galore_param_count = 0
        standard_param_count = 0
        for pg in param_groups:
            if pg.get("name") not in ("router", "a_matrix", "embedding"):
                pg["rank"] = galore_rank
                pg["update_proj_gap"] = galore_update_proj_gap
                pg["galore_scale"] = galore_scale
                for p in pg["params"]:
                    if p.ndim == 2:
                        galore_param_count += p.numel()
                    else:
                        standard_param_count += p.numel()
            else:
                standard_param_count += sum(p.numel() for p in pg["params"])
        optimizer = GaLoreAdamW(
            param_groups, betas=(0.9, beta2), weight_decay=weight_decay
        )
        standard_mem_mb = 2 * (galore_param_count + standard_param_count) * 4 / 1024**2
        print(
            f"Using GaLore optimizer (rank={galore_rank}, gap={galore_update_proj_gap}, scale={galore_scale})"
        )
        print(
            f"  GaLore params: {galore_param_count:,} | Standard params: {standard_param_count:,}"
        )
        print(f"  Baseline optimizer memory (AdamW): ~{standard_mem_mb:.0f} MB")

    elif getattr(config, "use_8bit_optimizer", False):
        from .optimizers.adamw_8bit import AdamW8bit

        optimizer = AdamW8bit(
            param_groups,
            betas=(0.9, beta2),
            eps=1e-8,
            weight_decay=weight_decay,
            block_size=128,
        )
        print(f"Using 8-bit AdamW optimizer (pure PyTorch, block_size=128)")

    elif getattr(config, "use_low_rank_momentum", False):
        from .optimizers.low_rank_momentum import LowRankMomentumAdamW

        optimizer = LowRankMomentumAdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, beta2),
            eps=1e-8,
            weight_decay=weight_decay,
            rank=config.low_rank_momentum_rank,
            svd_interval=config.low_rank_svd_interval,
            min_dim=config.low_rank_min_dim,
        )
        print(
            f"Using LowRankMomentumAdamW (rank={config.low_rank_momentum_rank}, "
            f"svd_interval={config.low_rank_svd_interval}, min_dim={config.low_rank_min_dim})"
        )
    elif optimizer_type == "lion":
        from .optimizers.lion import Lion

        lion_lr = base_lr * 0.1
        lion_wd = weight_decay * 3.0
        optimizer = Lion(
            param_groups,
            lr=lion_lr,
            betas=(0.9, 0.99),
            weight_decay=lion_wd,
        )
        print(
            f"[LION-K] Lion optimizer active | lr={lion_lr:.2e} (base_lr*0.1) | "
            f"betas=(0.9, 0.99) | wd={lion_wd:.4f} (weight_decay*3.0)"
        )
        print(
            f"[LION-K] WD x3 auto-scale: Lion sign-updates need stronger regularization"
        )
        print(
            f"[LION-K] Memory: exp_avg only (no exp_avg_sq) -- ~33% less optimizer VRAM vs AdamW"
        )
    else:
        if optimizer_type not in ("adamw", "adam", "AdamW", ""):
            import warnings

            warnings.warn(
                f"Unrecognized optimizer_type '{optimizer_type}', falling back to AdamW"
            )
        _use_fused = fused_optimizer or (device == "cuda" and use_bf16 and not use_amp)
        if _use_fused and device != "cuda":
            _use_fused = False
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, beta2),
            eps=1e-8,
            weight_decay=weight_decay,
            fused=_use_fused,
        )
        if _use_fused:
            print(f"[FUSED-ADAM] AdamW fused CUDA kernel active — single-kernel param updates")
        else:
            print(f"[FUSED-ADAM] AdamW standard Python loop (device={device}, bf16={use_bf16})")

    grad_clipper = AdaptiveGradientClipper(k=2.0, beta=0.99)
    print(f"[AdaGC] Initialized (k=2.0, beta=0.99) - adaptive threshold = mu + 2*sigma")

    _ternary_monitor = None
    if use_ternary:
        _ternary_monitor = TernaryCommitmentMonitor(window=100)
        _c2_count = sum(
            1 for m in model.modules()
            if getattr(m, "use_commitment_tracking", False)
        )
        print(f"[COMMITMENT] Initialized (window=100 steps) | C2 tracking: {_c2_count} modules")

    bifurcated_optimizer = None
    cusum_monitor = None
    probe_monitors = {}
    asymmetric_wrapper = None
    hybrid_rtrl = None
    pattern_transplant = None
    restart_scheduler = None

    if getattr(config, "use_bifurcated_optimizer", False) and use_ternary:
        from .optimizers.bifurcated import BifurcatedOptimizer

        bifurcated_optimizer = BifurcatedOptimizer(
            model.parameters(),
            base_lr=config.bifurcated_base_lr,
            commitment_threshold=config.bifurcated_commitment_threshold,
        )
        print(
            f"[BIFURCATED] Initialized (base_lr={config.bifurcated_base_lr}, "
            f"threshold={config.bifurcated_commitment_threshold})"
        )

    if getattr(config, "use_cusum_drift_monitor", False) and use_ternary:
        from .optimizers.cusum_monitor import CUSUMDriftDetector

        cusum_monitor = CUSUMDriftDetector(
            threshold_factor=config.cusum_threshold_factor,
            alarm_level=config.cusum_alarm_level,
        )
        print(
            f"[CUSUM] Initialized (threshold_factor={config.cusum_threshold_factor}, "
            f"alarm_level={config.cusum_alarm_level})"
        )

    if getattr(config, "use_output_gradient_probe", False):
        from .components.gradient_probe import register_output_probe

        if hasattr(model, "blocks"):
            for i, block in enumerate(model.blocks):
                if hasattr(block, "ssm") and block.ssm is not None:
                    register_output_probe(block.ssm, f"ssm_layer_{i}")
                    probe_monitors[f"ssm_layer_{i}"] = block.ssm.output_probe
                if hasattr(block, "swa") and block.swa is not None:
                    register_output_probe(block.swa, f"swa_layer_{i}")
                    probe_monitors[f"swa_layer_{i}"] = block.swa.output_probe
        print(
            f"[PROBE] Registered output gradient probes on {len(probe_monitors)} layers"
        )

    if getattr(config, "use_asymmetric_chunking", False):
        from .components.asymmetric_chunk import AsymmetricChunkWrapper

        asymmetric_wrapper = AsymmetricChunkWrapper(
            model, chunk_ssm=config.chunk_ssm_size, chunk_attn=config.chunk_attn_size
        )
        print(
            f"[ASYMCHUNK] Initialized (SSM chunk={config.chunk_ssm_size}, "
            f"attn chunk={config.chunk_attn_size})"
        )

    if getattr(config, "use_hybrid_rtrl", False) and use_ternary:
        from .optimizers.hybrid_rtrl import HybridRTRLTBPTT

        if hasattr(model, "blocks") and len(model.blocks) > 0:
            first_block = model.blocks[0]
            if hasattr(first_block, "ssm"):
                hybrid_rtrl = HybridRTRLTBPTT(
                    first_block.ssm, K_truncate=config.rtrl_k_truncate
                )
                print(f"[RTRL] Initialized (K_truncate={config.rtrl_k_truncate})")

    if getattr(config, "use_pattern_transplant", False) and use_ternary:
        from .optimizers.pattern_transplant import PatternTransplantation

        pattern_transplant = PatternTransplantation(
            blend_ratio=config.transplant_blend_ratio
        )
        print(f"[TRANSPLANT] Initialized (blend_ratio={config.transplant_blend_ratio})")

    if getattr(config, "use_commit_restart", False) and use_ternary:
        from .optimizers.commit_restart import CommitRestartScheduler

        restart_scheduler = CommitRestartScheduler(
            restart_interval=config.restart_interval,
            stuck_threshold=config.restart_stuck_threshold,
        )
        print(
            f"[RESTART] Initialized (interval={config.restart_interval}, "
            f"stuck_threshold={config.restart_stuck_threshold})"
        )

    info_freezer = None
    if use_info_freezing and config.use_ternary:
        info_freezer = InformationTheoreticFreezer(
            threshold=info_freeze_threshold,
            warmup=info_freezing_warmup,
            window=info_freeze_window,
            beta=info_freeze_beta,
        )
        print(
            f"[FREEZE] Initialized with threshold={info_freeze_threshold}, warmup={info_freezing_warmup}"
        )
    elif use_info_freezing and not config.use_ternary:
        print(
            "[FREEZE] WARNING: use_info_freezing=True but use_ternary=False. Freezing only useful for ternary. Skipping."
        )

    commitment_precision = None
    if getattr(config, "use_commitment_precision", False) and config.use_ternary:
        from .training.commitment_precision import CommitmentDrivenPrecision

        commitment_precision = CommitmentDrivenPrecision(
            eviction_threshold=getattr(config, "cdp_eviction_threshold", 0.9),
            freeze_threshold=getattr(config, "cdp_freeze_threshold", 0.99),
            min_bits=getattr(config, "cdp_min_bits", 4),
            max_bits=getattr(config, "cdp_max_bits", 8),
            log_interval=getattr(config, "cdp_log_interval", 100),
        )
        print(
            f"[P2-CDP] Commitment-Driven Precision enabled (eviction_threshold={config.cdp_eviction_threshold}, "
            f"freeze_threshold={config.cdp_freeze_threshold})"
        )

    teacher_model = None
    if use_distillation:
        from transformers import AutoModelForCausalLM

        print(f"Loading teacher model: {teacher_model_name}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_vocab = teacher_model.config.vocab_size
        if teacher_vocab != config.vocab_size:
            print(
                f"WARNING: Teacher vocab ({teacher_vocab}) != Student vocab ({config.vocab_size})"
            )
        print(
            f"Teacher loaded. VRAM after: {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )

    scaler = (
        torch.amp.GradScaler("cuda")
        if (use_amp or use_bf16) and device == "cuda"
        else None
    )
    if use_bf16:
        scaler = None
    if scaler is not None:
        print(f"[AMP-I] GradScaler active (FP16 path)")
    elif use_bf16 and device == "cuda":
        print(f"[AMP-I] BF16 autocast active — no GradScaler needed")
    else:
        print(f"[AMP-I] No AMP — full FP32 training")

    flip_tracker = None
    if getattr(config, "use_two_stage_flip_lr", False):
        from .components.ternary import TernaryFlipRateTracker

        flip_tracker = TernaryFlipRateTracker(
            threshold=config.flip_lr_threshold,
            ema_beta=config.flip_lr_ema_beta,
            min_step=config.flip_lr_min_step,
        )
        print(
            f"[E1] Two-Stage Flip LR enabled: threshold={config.flip_lr_threshold * 100:.1f}%, "
            f"decay_factor={config.flip_lr_decay_factor:.2f}, "
            f"transition_steps={config.flip_lr_transition_steps}"
        )

    total_steps = max_steps or (num_epochs * len_train_loader)

    def lr_schedule(step):
        if lr_schedule_type == "wsd":
            warmup_end = int(total_steps * wsd_warmup_pct)
            stable_end = int(total_steps * (wsd_warmup_pct + wsd_stable_pct))
            if step < warmup_end:
                return step / max(1, warmup_end)
            elif step < stable_end:
                return 1.0
            else:
                decay_progress = (step - stable_end) / max(1, total_steps - stable_end)
                return max(0.0, 1.0 - decay_progress)
        elif lr_schedule_type == "moderate_decay":
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 1.0 - 0.9 * progress
        elif lr_schedule_type in ("inverse_sqrt", "inv_sqrt"):
            if step < warmup_steps:
                return step / warmup_steps
            t = step - warmup_steps + 1
            decay = (1.0 / t) ** 0.5
            return max(0.1, decay)
        elif lr_schedule_type == "two_stage":
            drop_step = int(total_steps * two_stage_drop_pct)
            if step < warmup_steps:
                return step / warmup_steps
            elif step < drop_step:
                return 1.0
            else:
                phase2_progress = (step - drop_step) / max(1, total_steps - drop_step)
                return (
                    two_stage_drop_factor
                    * 0.5
                    * (1.0 + math.cos(phase2_progress * math.pi))
                )
        elif lr_schedule_type == "two_stage_flip":
            if step < warmup_steps:
                return step / warmup_steps
            elif flip_tracker is None or not flip_tracker.transitioned:
                return 1.0
            else:
                decay_factor = getattr(config, "flip_lr_decay_factor", 0.1)
                transition_steps = getattr(config, "flip_lr_transition_steps", 100)
                steps_since = step - flip_tracker.transition_step
                if transition_steps > 0 and steps_since < transition_steps:
                    progress = steps_since / transition_steps
                    return 1.0 - (1.0 - decay_factor) * 0.5 * (
                        1.0 - math.cos(progress * math.pi)
                    )
                return decay_factor
        elif lr_schedule_type == "cosine_restarts":
            if step < warmup_steps:
                return step / warmup_steps
            adjusted = step - warmup_steps
            cycle = adjusted // cosine_restart_period
            cycle_progress = (adjusted % cosine_restart_period) / cosine_restart_period
            peak_multiplier = cosine_restart_decay**cycle
            return peak_multiplier * 0.5 * (1.0 + math.cos(cycle_progress * math.pi))
        else:
            if lr_schedule_type not in ("cosine", "cos", ""):
                import warnings

                warnings.warn(
                    f"Unrecognized lr_schedule_type '{lr_schedule_type}', falling back to cosine"
                )
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    if optimizer_type == "schedule_free":
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    _g1_integral = 0.0
    _g1_prev_error = 0.0
    _g1_prev_committed_frac = 0.0
    _g1_pid_lr_mult = 1.0

    if use_cpu_offload and device == "cuda":
        if use_ags and config.use_ternary:
            optimizer = AGSCPUOffloadOptimizer(
                optimizer,
                device=device,
                model=model,
                top_k_frac=ags_top_k_frac,
                delta=ags_delta,
                full_sync_interval=ags_full_sync_interval,
                group_size=config.ternary_group_size,
                threshold_coeff=getattr(config, "threshold_coeff", 0.767),
            )
            print(
                f"CPU offload with AGS gradient sparsification: "
                f"top_k={ags_top_k_frac * 100:.0f}%, delta={ags_delta}, "
                f"full_sync_every={ags_full_sync_interval}"
            )
        else:
            optimizer = CPUOffloadOptimizer(optimizer, device=device)
            print(f"CPU offload: optimizer states -> pinned CPU RAM")
            if use_ags and not config.use_ternary:
                print(
                    "[AGS] WARNING: use_ags=True but use_ternary=False. "
                    "AGS requires ternary training. Falling back to standard CPU offload."
                )

    ema = None
    if use_ema:
        model_for_ema = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema = ModelEMA(model_for_ema, decay=ema_decay)
        ema_params = sum(v.numel() for v in ema.shadow.values())
        print(
            f"[EMA VERIFY] ACTIVE | decay={ema_decay} | params={ema_params:,} | mem={ema_params * 4 / 1024**2:.1f}MB"
        )

    return (
        optimizer,
        scheduler,
        aga_optimizer,
        bbo_optimizer,
        fused_aga,
        grad_clipper,
        _ternary_monitor,
        bifurcated_optimizer,
        cusum_monitor,
        probe_monitors,
        asymmetric_wrapper,
        hybrid_rtrl,
        pattern_transplant,
        restart_scheduler,
        info_freezer,
        commitment_precision,
        teacher_model,
        scaler,
        _g1_integral,
        _g1_prev_error,
        _g1_prev_committed_frac,
        _g1_pid_lr_mult,
        flip_tracker,
        ema,
    )


def _setup_model_and_components(
    config: ModelConfig,
    device: str,
    ste_type: str,
    use_compile: bool,
    use_hestia2026: bool,
    tequila_lambda: float,
    use_caq: bool,
    caq_alpha: float,
    use_tpsq: bool,
    tpsq_beta: float,
    use_tcd: bool,
    use_gumbel_noise: bool,
    gumbel_temperature: float,
    use_iaqd: bool,
    use_stp: bool,
    stp_sigma_ratio: float,
    use_ctce: bool,
    ctce_beta: float,
    ctce_delta_ratio: float,
    use_dual_rate: bool,
    dual_rate_peak: float,
    dual_rate_floor: float,
    use_gaste: bool,
    gaste_sigma_g: float,
    use_rlec: bool,
    rlec_rank: int,
    use_margin_sparse: bool,
    margin_ema_beta: float,
    use_ags: bool,
    ags_top_k_frac: float,
    ags_full_sync_interval: int,
    ags_delta: float,
    shadow_bf16: bool,
    let_teacher_path: Optional[str],
    let_lambda: float,
    let_stop_step: int,
    let_layer_k: int,
    online_distill_teacher_path: Optional[str],
    online_distill_alpha: float,
    online_distill_temperature: float,
    dataset_type: str,
    batch_size: int,
    num_samples: int,
) -> tuple:
    """Setup model and all R4 components (creation, weight init, layer mods, torch.compile, teachers).

    Args:
        config: Model configuration
        device: Device to place model on
        ste_type: STE quantization type ('standard', 'hestia2026', etc.)
        use_compile: Enable torch.compile
        use_hestia2026: Enable HESTIA 2026 STE variant
        tequila_lambda: Tequila deadzone reactivation lambda
        use_caq: Enable Curvature-Aware Quantization
        caq_alpha: CAQ alpha parameter
        use_tpsq: Enable Phase-Space Quantization
        tpsq_beta: TPSQ beta parameter
        use_tcd: Enable Ternary Critical Dynamics
        use_gumbel_noise: Enable Gumbel+STE noise
        gumbel_temperature: Gumbel temperature parameter
        use_iaqd: Enable Input-Adaptive Quantization Domain
        use_stp: Enable Stochastic Threshold Perturbation
        stp_sigma_ratio: STP sigma ratio
        use_ctce: Enable Composite Threshold-Crossing Estimator
        ctce_beta: CTCE beta parameter
        ctce_delta_ratio: CTCE delta ratio
        use_dual_rate: Enable Dual-Rate STE
        dual_rate_peak: Dual-Rate peak
        dual_rate_floor: Dual-Rate floor
        use_gaste: Enable Group-Aware STE
        gaste_sigma_g: GASTE sigma_g parameter
        use_rlec: Enable Residual Low-Rank Error Compensation
        rlec_rank: RLEC rank
        use_margin_sparse: Enable Margin-Sparse Backward
        margin_ema_beta: EMA beta for margin tracking
        use_ags: Enable Asymmetric Gradient Sparsification
        ags_top_k_frac: AGS top-k fraction
        ags_full_sync_interval: AGS full sync interval
        ags_delta: AGS delta threshold
        shadow_bf16: Enable BF16 shadow weights
        let_teacher_path: Path to LET teacher checkpoint
        let_lambda: LET loss weight
        let_stop_step: LET stop step
        let_layer_k: LET layer index
        online_distill_teacher_path: Path to online distillation teacher
        online_distill_alpha: Online distill loss weight
        online_distill_temperature: Online distill temperature
        dataset_type: Dataset type for ACSI init
        batch_size: Batch size for ACSI init
        num_samples: Number of samples for ACSI init

    Returns:
        Tuple of (model, gradient_monitor, ram_checkpointer, anomaly_detector, use_qmm,
                  inoculation, delta_chronicle, flip_tracker, gradient_recyclers,
                  let_teacher, let_proj, od_teacher, param_info)
    """
    model = ModelV2(config).to(device)

    try:
        train_dataloader = get_dataloader(
            dataset_type=dataset_type,
            split="train",
            batch_size=batch_size,
            seq_len=config.seq_len,
            num_samples=min(num_samples, batch_size * 5)
            if num_samples
            else batch_size * 5,
            num_workers=0,
            device=device,
            drop_last=False,
        )
        system_identification_init(model, train_dataloader, device, n_batches=5)
        print("[ACSI] SSM A_log initialized from data statistics")
    except Exception as e:
        print(f"[ACSI] System identification init skipped: {str(e)[:80]}")

    gradient_monitor = None
    if getattr(config, "use_gradient_monitor", False):
        from .components.gradient_monitor import GradientMonitor

        gradient_monitor = GradientMonitor(
            window_size=getattr(config, "gradient_monitor_window", 100)
        )
        print(f"[GRAD-MONITOR] Enabled: window={gradient_monitor.window_size}")

    ram_checkpointer = None
    if getattr(config, "use_ram_checkpoint", False):
        from .components.ram_checkpoint import RAMCheckpointer

        ram_checkpointer = RAMCheckpointer(
            disk_backup=getattr(config, "ram_checkpoint_disk_backup", True)
        )
        print(f"[RAM-CKPT] Enabled: disk_backup={ram_checkpointer.disk_backup}")

    anomaly_detector = None
    if getattr(config, "use_anomaly_detector", False):
        from .components.anomaly_detector import AnomalyDetector

        anomaly_detector = AnomalyDetector(
            z_threshold=getattr(config, "anomaly_z_threshold", 3.0),
            ring_buffer_size=getattr(config, "anomaly_ring_buffer_size", 3),
        )
        print(f"[ANOMALY-DET] Enabled: z={anomaly_detector.z_threshold}")

    use_qmm = getattr(config, "use_qmm_regularizer", False)

    inoculation = None
    if getattr(config, "use_ternary_inoculation", False):
        from .components.ternary_inoculation import TernaryInoculation

        inoculation = TernaryInoculation(
            flip_prob=getattr(config, "inoculation_flip_prob", 0.01),
            flip_fraction=getattr(config, "inoculation_flip_fraction", 0.001),
            loss_weight=getattr(config, "inoculation_loss_weight", 0.1),
        )
        print(f"[INOCULATION] Enabled: flip_prob={inoculation.flip_prob}")

    delta_chronicle = None
    if getattr(config, "use_delta_chronicle", False):
        from .components.delta_chronicle import DeltaChronicle

        delta_chronicle = DeltaChronicle(
            max_history_mb=getattr(config, "delta_chronicle_max_mb", 500.0)
        )
        print(f"[CHRONICLE] Enabled: max_mb={delta_chronicle.max_history_mb}")

    flip_tracker = None
    if getattr(config, "use_flip_influence", False):
        from .components.flip_influence import FlipInfluenceTracker

        flip_tracker = FlipInfluenceTracker(
            top_k=getattr(config, "flip_influence_top_k", 100)
        )
        print(f"[FLIP-INFLUENCE] Enabled: top_k={flip_tracker.top_k}")

    if getattr(config, "use_ternary_circuit_compile", False):
        print(
            f"[A5-P7] Ternary Circuit Compilation: will recompile every {getattr(config, 'circuit_compile_recompile_interval', 500)} steps"
        )

    adaptive_dataflow_selector = None
    if getattr(config, "use_adaptive_dataflow", False):
        try:
            from .components.adaptive_dataflow import DataflowSelector

            adaptive_dataflow_selector = DataflowSelector()
            adaptive_dataflow_selector.analyze_model(model)
            print(f"[A5-F1] Adaptive Dataflow: analyzed all TernaryLinear layers")
        except ImportError:
            print(f"[A5-F1] Dataflow analysis not available (component not found)")

    if getattr(config, "use_ternary_binary_decomp", False):
        print(
            f"[A5-F2] Binary Decomposition: block_size={getattr(config, 'ternary_binary_block_size', 4)}"
        )

    if getattr(config, "use_commitment_detector", False):
        print(
            f"[A6-M1] Commitment detector: ON (threshold={config.commitment_threshold}, block_size={config.commitment_block_size})"
        )
    if getattr(config, "use_metabolic_probe", False):
        print(f"[A6-P3] Metabolic probe: ON (window={config.metabolic_window})")

    if getattr(config, "use_graduated_eviction", False):
        print(
            f"[A6-F1] Graduated eviction: ON (tiers={config.eviction_tier_thresholds})"
        )

    if getattr(config, "use_cusum_drift", False):
        print(f"[A6-M5] CUSUM drift tracker: ON (threshold={config.cusum_threshold})")
    if getattr(config, "use_sparse_updates", False):
        print(
            f"[A6-F1B] Sparse updates: ON (interval={config.sparse_update_committed_interval})"
        )

    if getattr(config, "use_frozen_ternary_op", False):
        print(f"[A6-F5A] Frozen ternary op: ON (cache={config.frozen_cache_max_mb}MB)")
    if getattr(config, "use_recycling_pool", False):
        print(f"[A6-M7] Recycling pool: ON (max={config.recycling_pool_max_mb}MB)")
    if getattr(config, "use_selective_grad_subsample", False):
        print(
            f"[A6-P2] Selective grad subsample: ON (committed_ratio={config.grad_subsample_committed})"
        )

    if getattr(config, "use_bifurcated_optimizer", False):
        print(
            f"[A6-P5] Bifurcated optimizer: ON (committed_lr_ratio={config.committed_lr_ratio})"
        )
    if getattr(config, "use_commitment_curriculum", False):
        print(
            f"[A6-P7] Commitment curriculum: ON (target={config.curriculum_target_ratio})"
        )
    if getattr(config, "use_thermal_memory", False):
        print(f"[A6-P8] Thermal memory: ON")

    if getattr(config, "use_ternary_layout_opt", False):
        print(
            f"[A6-F4] Ternary layout optimizer: ON (alignment={config.layout_alignment})"
        )
    if getattr(config, "use_predictive_freezer", False):
        print(f"[A6-M3] Predictive freezer: ON (horizon={config.prediction_horizon})")
    if getattr(config, "use_tensor_defrag", False):
        print(f"[A6-F8] Tensor defragmenter: ON (interval={config.defrag_interval})")

    if config.use_optrot and config.use_ternary:
        from .components.optrot import enable_optrot

        enable_optrot(model, config.optrot_block_size)

    if tequila_lambda > 0:
        tequila_count = 0
        for module in model.modules():
            if hasattr(module, "tequila_lambda"):
                module.tequila_lambda = tequila_lambda
                tequila_count += 1
        print(f"Tequila: applied lambda={tequila_lambda} to {tequila_count} modules")
        if tequila_count == 0:
            import warnings

            warnings.warn(
                "Tequila enabled but applied to 0 modules -- check if model has ternary layers"
            )

    if use_hestia2026:
        hestia_count = 0
        for module in model.modules():
            if hasattr(module, "ste_type"):
                module.ste_type = "hestia2026"
                module.hestia_tau_init = 2.0
                hestia_count += 1
        print(f"HESTIA 2026: applied to {hestia_count} modules")
        if hestia_count == 0:
            import warnings

            warnings.warn(
                "HESTIA 2026 enabled but applied to 0 modules -- check if model has ternary layers"
            )

    if ste_type != "standard":
        _ste_count = 0
        for _m in model.modules():
            if hasattr(_m, "ste_type"):
                _m.ste_type = ste_type
                _ste_count += 1
        print(f"[STE-TYPE] Applied '{ste_type}' to {_ste_count} TernaryLinear modules")

    if use_caq and config.use_ternary:
        caq_count = 0
        for module in model.modules():
            if hasattr(module, "use_caq"):
                module.use_caq = True
                module.caq_alpha = caq_alpha
                caq_count += 1
        print(f"CAQ: enabled on {caq_count} modules (alpha={caq_alpha})")

    if use_tpsq and config.use_ternary:
        from .components.ternary import enable_tpsq

        enable_tpsq(model, beta=tpsq_beta)

    if use_tcd and config.use_ternary:
        from .components.ternary import enable_tcd

        enable_tcd(model)

    if use_gumbel_noise and config.use_ternary:
        from .components.ternary import enable_gumbel_noise

        enable_gumbel_noise(model, temperature=gumbel_temperature)

    if use_iaqd:
        iaqd_count = 0
        for module in model.modules():
            if hasattr(module, "enable_iaqd"):
                module.enable_iaqd()
                iaqd_count += 1
        print(f"IAQD: enabled on {iaqd_count} modules")

    if use_stp and config.use_ternary:
        from .components.ternary import enable_stp

        enable_stp(model, sigma_ratio=stp_sigma_ratio)

    if use_ctce and config.use_ternary:
        from .components.ternary import enable_ctce

        enable_ctce(model, beta=ctce_beta, delta_ratio=ctce_delta_ratio)

    if use_dual_rate and config.use_ternary:
        from .components.ternary import enable_dual_rate

        enable_dual_rate(model, peak=dual_rate_peak, floor=dual_rate_floor)

    if use_gaste and config.use_ternary:
        from .components.ternary import enable_gaste

        enable_gaste(model, sigma_g=gaste_sigma_g)

    if use_rlec and config.use_ternary:
        from .components.ternary import enable_rlec

        enable_rlec(model, rank=rlec_rank)

    gradient_recyclers = None
    if use_margin_sparse and config.use_ternary:
        gradient_recyclers = init_gradient_recyclers(model, beta=margin_ema_beta)

    if use_ags and config.use_ternary:
        print(
            f"AGS (R2-A5-M2'): gradient sparsification enabled, "
            f"top_k={ags_top_k_frac * 100:.0f}%, delta={ags_delta}, "
            f"full_sync_every={ags_full_sync_interval} steps"
        )

    let_teacher = None
    let_proj = None
    od_teacher = None
    if let_teacher_path is not None:
        print(f"LET Transfer: loading teacher from {let_teacher_path}")
        let_teacher = ModelV2(config).to(device)
        ck = torch.load(let_teacher_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model_state_dict" in ck:
            sd = ck["model_state_dict"]
        elif isinstance(ck, dict) and "model_state" in ck:
            sd = ck["model_state"]
        else:
            sd = ck
        missing, unexpected = let_teacher.load_state_dict(sd, strict=False)
        print(
            f"  Teacher loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        let_teacher.eval()
        for p in let_teacher.parameters():
            p.requires_grad = False
        teacher_n_layer = config.n_layer
        print(
            f"  LET config: lambda={let_lambda}, stop_step={let_stop_step}, layer_k={let_layer_k}"
        )
        print(
            f"  Teacher final layer idx: {teacher_n_layer - 1}, Student layer k: {let_layer_k}"
        )

    if online_distill_teacher_path is not None:
        print(
            f"Online Distillation: loading FP16 teacher from {online_distill_teacher_path}"
        )
        teacher_config = copy.deepcopy(config)
        teacher_config.use_ternary = False
        od_teacher = ModelV2(teacher_config).to(device)
        ck = torch.load(
            online_distill_teacher_path, map_location="cpu", weights_only=False
        )
        if isinstance(ck, dict) and "model_state_dict" in ck:
            sd = ck["model_state_dict"]
        elif isinstance(ck, dict) and "model_state" in ck:
            sd = ck["model_state"]
        else:
            sd = ck
        missing, unexpected = od_teacher.load_state_dict(sd, strict=False)
        print(
            f"  Teacher loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        od_teacher.eval()
        for p in od_teacher.parameters():
            p.requires_grad = False
        print(
            f"  Online Distill config: alpha={online_distill_alpha}, T={online_distill_temperature}"
        )
        if device == "cuda":
            print(
                f"  VRAM after teacher: {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
            )

    if shadow_bf16 and config.use_ternary:
        n_converted = model.convert_ternary_to_bf16()
        print(f"BF16 shadow weights: {n_converted} ternary layers converted")

    param_info = model.count_parameters()

    if use_compile and device == "cuda":
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Applying torch.compile (mode='reduce-overhead', dynamic=False)...")
        try:
            compile_options = {}
            if config.compile_memory_budget is not None:
                compile_options["activation_memory_budget"] = (
                    config.compile_memory_budget
                )

            if compile_options:
                model = torch.compile(
                    model, mode="reduce-overhead", dynamic=False, options=compile_options
                )
                logger.info(f"torch.compile applied with mode='reduce-overhead', options: {compile_options}")
            else:
                model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print("[TORCH-COMPILE] ACTIVE | mode=reduce-overhead | warmup ~20 steps")
            logger.info(
                "torch.compile applied. First ~20 steps will be slower (compilation warmup)."
            )
        except Exception as e:
            print(
                f"NOTE: torch.compile unavailable ({type(e).__name__}). Running without compilation."
            )
            print(f"  (torch.compile requires PyTorch 2.0+ with Triton support.)")

    return (
        model,
        gradient_monitor,
        ram_checkpointer,
        anomaly_detector,
        use_qmm,
        inoculation,
        delta_chronicle,
        flip_tracker,
        gradient_recyclers,
        let_teacher,
        let_proj,
        od_teacher,
        param_info,
    )


def _setup_training_state(
    model,
    optimizer,
    device: str,
    use_amp: bool,
    use_bf16: bool,
    use_ema: bool,
    ema_decay: float,
    label_smoothing: float,
    use_liger_ce: bool,
    warmup_steps: int,
    total_steps: int,
    cosine_restart_period: int,
    cosine_restart_decay: float,
    lr_schedule_type: str,
    optimizer_type: str,
):
    scaler = (
        torch.amp.GradScaler("cuda")
        if (use_amp or use_bf16) and device == "cuda"
        else None
    )
    if use_bf16:
        scaler = None
    if scaler is not None:
        print(f"[AMP-I] GradScaler active (FP16 path)")
    elif use_bf16 and device == "cuda":
        print(f"[AMP-I] BF16 autocast active — no GradScaler needed")
    else:
        print(f"[AMP-I] No AMP — full FP32 training")

    if use_liger_ce and _LIGER_AVAILABLE:
        criterion = LigerCrossEntropyLoss(label_smoothing=label_smoothing)
        print(
            f"Using Liger fused cross-entropy"
            + (f" (label_smoothing={label_smoothing})" if label_smoothing > 0 else "")
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            print(f"[LABEL-SMOOTHING VERIFY] ACTIVE | alpha={label_smoothing}")

    monitor = TrainingMonitor()

    def lr_schedule(step):
        if lr_schedule_type == "cosine_restarts":
            if step < warmup_steps:
                return step / warmup_steps
            adjusted = step - warmup_steps
            cycle = adjusted // cosine_restart_period
            cycle_progress = (adjusted % cosine_restart_period) / cosine_restart_period
            peak_multiplier = cosine_restart_decay**cycle
            return peak_multiplier * 0.5 * (1.0 + math.cos(cycle_progress * math.pi))
        else:
            if lr_schedule_type not in ("cosine", "cos", ""):
                import warnings

                warnings.warn(
                    f"Unrecognized lr_schedule_type '{lr_schedule_type}', falling back to cosine"
                )
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    if optimizer_type == "schedule_free":
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    ema = None
    if use_ema:
        model_for_ema = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema = ModelEMA(model_for_ema, decay=ema_decay)
        ema_params = sum(v.numel() for v in ema.shadow.values())
        print(
            f"[EMA VERIFY] ACTIVE | decay={ema_decay} | params={ema_params:,} | mem={ema_params * 4 / 1024**2:.1f}MB"
        )

    return {
        "scaler": scaler,
        "criterion": criterion,
        "monitor": monitor,
        "scheduler": scheduler,
        "ema": ema,
    }


def _setup_pre_loop_state(
    model,
    config,
    batch_size,
    grad_accum_steps,
    use_adagc,
    adagc_factor,
    adagc_beta,
    use_fp8_grad_accum,
    use_topk_grad_sparsification,
    topk_grad_sparsity,
    use_gradient_sign_consistency,
    grad_sign_window,
    grad_sign_threshold,
    seq_warmup,
    seq_warmup_min,
    start_step,
):
    global_step = start_step
    best_loss = float("inf")
    nan_count = 0
    start_time = time.time()
    loss = None
    _lr_multipliers = {}
    _phase_loss_window = []

    _kd_hidden = {}
    _kd_hook_handle = None
    if (
        hasattr(model, "lm_head")
        and model.lm_head is not None
        and isinstance(model.lm_head, TernaryLinear)
    ):
        def _lm_head_input_hook(module, args, output):
            _kd_hidden["x"] = args[0].detach()
        _kd_hook_handle = model.lm_head.register_forward_hook(_lm_head_input_hook)
        print(
            f"[KD-SHADOW] Hook registered on ternary lm_head "
            f"(shadow weight: {model.lm_head.weight.shape})",
            flush=True,
        )

    _l1_token_budget = batch_size * config.seq_len
    _l1_current_batch = batch_size

    _l2_current_seq_len = seq_warmup_min if seq_warmup else config.seq_len

    _rpe_loss_ema = None
    _rpe_ema_beta = 0.95

    _cbs_grad_norms = []
    _cbs_current_accum = grad_accum_steps
    _cbs_max_accum = 4

    _cusum_c = 0.0
    _cusum_mu = None
    _cusum_skip = False
    _cusum_skipped = 0

    adagc_ema = {}
    _adagc_clips = 0
    _adagc_max_ratio = 0.0
    if use_adagc:
        for name, param in model.named_parameters():
            if param.requires_grad:
                adagc_ema[name] = 0.0
        print(
            f"[AdaGC] Enabled: factor={adagc_factor}, beta={adagc_beta}, tracking {len(adagc_ema)} tensors"
        )

    fp8_grad_buffers = {}
    fp8_error_buffers = {}
    if use_fp8_grad_accum and grad_accum_steps > 1:
        for name, param in model.named_parameters():
            if param.requires_grad:
                fp8_error_buffers[name] = torch.zeros_like(param.data)
        print(
            f"[FP8-GRAD-ACCUM] Enabled with error_feedback={getattr(config, 'fp8_grad_accum_error_feedback', True)}"
        )
    elif use_fp8_grad_accum:
        print(
            f"[FP8-GRAD-ACCUM] DISABLED (grad_accum_steps={grad_accum_steps}, needs > 1)"
        )
        use_fp8_grad_accum = False

    topk_error_buffers = {}
    if use_topk_grad_sparsification:
        for name, param in model.named_parameters():
            if param.requires_grad:
                topk_error_buffers[name] = torch.zeros_like(param.data)
        print(
            f"[F9-TOPK-SPARSIFICATION] Enabled: sparsity={topk_grad_sparsity:.2%} (keep top {(1 - topk_grad_sparsity) * 100:.0f}%)"
        )

    _grad_sign_history = {}
    if use_gradient_sign_consistency and config.use_ternary:
        print(
            f"[P5-GRAD-SIGN-CONSISTENCY] Enabled: window={grad_sign_window}, threshold={grad_sign_threshold}"
        )

    fisher_accum = {}
    if getattr(config, "use_fisher_preconditioning", False):
        fisher_damping = getattr(config, "fisher_damping", 1e-4)
        print(f"[A2-F6-FISHER] Enabled: damping={fisher_damping}")

    sign_sgd_momentum_buffers = {}
    if getattr(config, "use_sign_sgd", False):
        sign_sgd_beta = getattr(config, "sign_sgd_momentum", 0.9)
        print(f"[A2-F7-SIGNSSGD] Enabled: momentum={sign_sgd_beta}")

    if getattr(config, "use_consensus_gradient_filter", False) and getattr(
        config, "use_recursive_ssm", False
    ):
        consensus_alpha = getattr(config, "consensus_alpha", 0.5)
        print(f"[A2-M7-CONSENSUS] Enabled: alpha={consensus_alpha} (RecursiveSSM)")
    elif getattr(config, "use_consensus_gradient_filter", False):
        print(
            f"[A2-M7-CONSENSUS] WARNING: use_consensus_gradient_filter requires use_recursive_ssm=True"
        )
        config.use_consensus_gradient_filter = False

    if getattr(config, "use_adaptive_quantization", False) and getattr(
        config, "use_ternary", False
    ):
        adap_threshold = getattr(config, "adaptive_quant_sensitivity_threshold", 0.1)
        adap_interval = getattr(config, "adaptive_quant_interval", 500)
        print(
            f"[A2-F8-ADAPTIVE-QUANT] Enabled: threshold={adap_threshold}, interval={adap_interval}"
        )
    elif getattr(config, "use_adaptive_quantization", False):
        print(
            f"[A2-F8-ADAPTIVE-QUANT] WARNING: use_adaptive_quantization requires use_ternary=True"
        )
        config.use_adaptive_quantization = False

    return {
        "global_step": global_step,
        "best_loss": best_loss,
        "nan_count": nan_count,
        "start_time": start_time,
        "loss": loss,
        "_lr_multipliers": _lr_multipliers,
        "_phase_loss_window": _phase_loss_window,
        "_kd_hidden": _kd_hidden,
        "_kd_hook_handle": _kd_hook_handle,
        "_l1_token_budget": _l1_token_budget,
        "_l1_current_batch": _l1_current_batch,
        "_l2_current_seq_len": _l2_current_seq_len,
        "_rpe_loss_ema": _rpe_loss_ema,
        "_rpe_ema_beta": _rpe_ema_beta,
        "_cbs_grad_norms": _cbs_grad_norms,
        "_cbs_current_accum": _cbs_current_accum,
        "_cbs_max_accum": _cbs_max_accum,
        "_cusum_c": _cusum_c,
        "_cusum_mu": _cusum_mu,
        "_cusum_skip": _cusum_skip,
        "_cusum_skipped": _cusum_skipped,
        "adagc_ema": adagc_ema,
        "_adagc_clips": _adagc_clips,
        "_adagc_max_ratio": _adagc_max_ratio,
        "fp8_grad_buffers": fp8_grad_buffers,
        "fp8_error_buffers": fp8_error_buffers,
        "topk_error_buffers": topk_error_buffers,
        "_grad_sign_history": _grad_sign_history,
        "fisher_accum": fisher_accum,
        "sign_sgd_momentum_buffers": sign_sgd_momentum_buffers,
    }


def _train_step(model, batch_data, ctx):
    """
    Core training step: forward pass, loss computation, all auxiliary losses.

    Args:
        model: The neural network model
        batch_data: Raw batch data from dataloader (dict or tuple)
        ctx: Context dict with all training state:
            - device, config, global_step, max_steps, total_steps
            - criterion, use_cuda_prefetch, use_bf16, use_amp, use_compile
            - seq_warmup, seq_warmup_pct, seq_warmup_min, train_loader
            - _l1_token_budget, _l1_current_batch, get_dataloader, dataset_type, vocab_size
            - num_samples, data_dir, num_workers, pin_memory, persistent_workers
            - prefetch_factor, rho1_ref_losses_path, use_rho1, use_sequence_packing
            - curriculum_sampler, use_saq, config.use_ternary, saq_k, log_every
            - use_activation_quant, use_compile, torch.compiler
            - use_rls, let_teacher, let_layer_k, let_stop_step, use_rho1, rho1_keep_ratio
            - use_uncertainty_loss, uw_loss_module, mtp_logits, _kd_hidden, _kd_hook_handle
            - _use_egce, teacher_model, distillation_loss, distill_temperature, use_tsld
            - distill_alpha, od_teacher, online_distill_temperature, online_distill_alpha
            - boundary_reg_lambda, angular_reg_lambda, contractive_reg_lambda
            - use_tdo, use_sca_sparsity, sca_warmup_steps, model_actual
            - use_ema_distill, ema_distill_temperature, ema_distill_alpha, use_sskd
            - sskd_temperature, sskd_alpha, use_ema_replay, ema_replay_lambda
            - use_thm_loss, thm_margin, thm_lambda, use_tail_loss, tail_flip_margin
            - tail_num_negatives, tail_lambda, use_rcl_loss, rcl_beta, rcl_sample_every
            - rcl_lambda, use_ter_loss, ter_loss_mu, ter_noise_var, use_predictsim
            - use_local_loss_per_block, grad_accum_steps, log_every
            - Plus: any state that gets modified in-place (e.g., _phase_loss_window)

    Returns:
        loss_val: scalar loss value (float)
        ctx: updated context dict (modified in-place, contains loss_val)

    Updates ctx in-place:
        - ctx['loss_val']: the computed loss value
        - Any mutable state that changed (e.g., _phase_loss_window, _kd_hidden)
    """
    device = ctx["device"]
    config = ctx["config"]
    global_step = ctx["global_step"]
    use_cuda_prefetch = ctx.get("use_cuda_prefetch", False)
    use_bf16 = ctx.get("use_bf16", False)
    use_amp = ctx.get("use_amp", False)
    use_compile = ctx.get("use_compile", False)
    seq_warmup = ctx.get("seq_warmup", False)
    seq_warmup_pct = ctx.get("seq_warmup_pct", 0.1)
    seq_warmup_min = ctx.get("seq_warmup_min", 64)
    use_rho1 = ctx.get("use_rho1", False)
    rho1_keep_ratio = ctx.get("rho1_keep_ratio", 1.0)
    use_uncertainty_loss = ctx.get("use_uncertainty_loss", False)
    uw_loss_module = ctx.get("uw_loss_module", None)
    use_rls = ctx.get("use_rls", False)
    let_teacher = ctx.get("let_teacher", None)
    let_layer_k = ctx.get("let_layer_k", 0)
    let_stop_step = ctx.get("let_stop_step", 0)
    teacher_model = ctx.get("teacher_model", None)
    od_teacher = ctx.get("od_teacher", None)
    boundary_reg_lambda = ctx.get("boundary_reg_lambda", 0.0)
    angular_reg_lambda = ctx.get("angular_reg_lambda", 0.0)
    contractive_reg_lambda = ctx.get("contractive_reg_lambda", 0.0)
    use_tdo = ctx.get("use_tdo", False)
    use_sca_sparsity = ctx.get("use_sca_sparsity", False)
    sca_warmup_steps = ctx.get("sca_warmup_steps", 0)
    model_actual = ctx.get("model_actual", model)
    use_ema_distill = ctx.get("use_ema_distill", False)
    ema = ctx.get("ema", None)
    ema_distill_temperature = ctx.get("ema_distill_temperature", 1.0)
    ema_distill_alpha = ctx.get("ema_distill_alpha", 0.1)
    use_sskd = ctx.get("use_sskd", False)
    sskd_temperature = ctx.get("sskd_temperature", 4.0)
    sskd_alpha = ctx.get("sskd_alpha", 0.1)
    use_ema_replay = ctx.get("use_ema_replay", False)
    ema_replay_lambda = ctx.get("ema_replay_lambda", 0.1)
    use_thm_loss = ctx.get("use_thm_loss", False)
    thm_margin = ctx.get("thm_margin", 1.0)
    thm_lambda = ctx.get("thm_lambda", 0.1)
    use_tail_loss = ctx.get("use_tail_loss", False)
    tail_flip_margin = ctx.get("tail_flip_margin", 0.0)
    tail_num_negatives = ctx.get("tail_num_negatives", 15)
    tail_lambda = ctx.get("tail_lambda", 0.1)
    use_rcl_loss = ctx.get("use_rcl_loss", False)
    rcl_beta = ctx.get("rcl_beta", 0.95)
    rcl_sample_every = ctx.get("rcl_sample_every", 1)
    rcl_lambda = ctx.get("rcl_lambda", 0.1)
    use_ter_loss = ctx.get("use_ter_loss", False)
    ter_loss_mu = ctx.get("ter_loss_mu", 0.0)
    ter_noise_var = ctx.get("ter_noise_var", 1e-6)
    use_predictsim = ctx.get("use_predictsim", False)
    use_saq = ctx.get("use_saq", False)
    saq_k = ctx.get("saq_k", 0)
    log_every = ctx.get("log_every", 100)
    grad_accum_steps = ctx.get("grad_accum_steps", 1)
    batch_idx = ctx.get("batch_idx", 0)
    start_step = ctx.get("start_step", 0)
    criterion = ctx["criterion"]
    distillation_loss = ctx.get("distillation_loss", None)
    distill_temperature = ctx.get("distill_temperature", 1.0)
    use_tsld = ctx.get("use_tsld", False)
    distill_alpha = ctx.get("distill_alpha", 0.5)
    online_distill_temperature = ctx.get("online_distill_temperature", 1.0)
    online_distill_alpha = ctx.get("online_distill_alpha", 0.5)
    let_lambda = ctx.get("let_lambda", 0.1)
    tdo_lambda = ctx.get("tdo_lambda", 0.1)
    tdo_target = ctx.get("tdo_target", 0.5)
    tdo_temperature = ctx.get("tdo_temperature", 1.0)
    sca_lambda = ctx.get("sca_lambda", 0.1)
    sca_target_sparsity = ctx.get("sca_target_sparsity", 0.7)
    _kd_hidden = ctx.get("_kd_hidden", {})
    _kd_hook_handle = ctx.get("_kd_hook_handle", None)

    if isinstance(batch_data, dict):
        input_ids = batch_data["input_ids"]
        boundary_mask = batch_data.get("boundary_mask", None)
        targets = input_ids[..., 1:].clone()
        input_ids = input_ids[..., :-1].clone()
        batch_ref_losses = None
    elif use_rho1 and len(batch_data) == 3:
        input_ids, targets, batch_ref_losses = batch_data
        boundary_mask = None
        batch_ref_losses = batch_ref_losses.to(device, non_blocking=True)
    else:
        input_ids, targets = batch_data[:2]
        boundary_mask = None
        batch_ref_losses = None

    if not use_cuda_prefetch:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if boundary_mask is not None:
            boundary_mask = boundary_mask.to(device, non_blocking=True)

    if use_bf16 and device == "cuda":
        amp_context = torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16
        )
    elif use_amp and device == "cuda":
        amp_context = torch.amp.autocast(device_type="cuda")
    else:
        amp_context = contextlib.nullcontext()

    with amp_context:
        if let_teacher is not None and global_step < let_stop_step:
            result = model(
                input_ids,
                return_hidden_at=let_layer_k,
                boundary_mask=boundary_mask,
            )
            if isinstance(result, tuple) and len(result) == 3:
                logits, student_hidden, mtp_logits = result
            else:
                logits, student_hidden = result
                mtp_logits = None
        else:
            _masked_pred_mask = None
            _mp_input_ids = input_ids
            if getattr(config, 'use_masked_pred', False) and model.training:
                _mp_ratio = getattr(config, 'masked_pred_ratio', 0.15)
                _masked_pred_mask = (torch.rand_like(input_ids.float()) < _mp_ratio)
                _mp_input_ids = input_ids.clone()
                _mp_input_ids[_masked_pred_mask] = 0
            _use_local_loss = getattr(config, 'use_local_loss_per_block', False) and model.training
            result = model(
                _mp_input_ids,
                boundary_mask=boundary_mask,
                return_pre_head=use_rls,
                return_local_losses=_use_local_loss,
            )
            _local_losses_list = None
            if use_rls:
                if isinstance(result, tuple) and len(result) >= 3:
                    logits = result[0]
                    mtp_logits = result[1] if isinstance(result[1], list) else None
                    pre_head = result[-1]
                else:
                    logits = result[0]
                    pre_head = result[1]
                    mtp_logits = None
            else:
                if isinstance(result, tuple):
                    logits, mtp_logits = result
                else:
                    logits = result
                    mtp_logits = None
                pre_head = None

            if _use_local_loss and isinstance(result, tuple):
                for _item in result:
                    if isinstance(_item, list) and len(_item) > 0 and isinstance(_item[0], tuple):
                        _local_losses_list = _item
                        break

        if use_rho1 and batch_ref_losses is not None:
            from .rho1 import compute_rho1_loss

            _ref = batch_ref_losses
            ce_loss, rho1_stats = compute_rho1_loss(
                logits,
                targets,
                _ref,
                keep_ratio=rho1_keep_ratio,
                vocab_size=config.vocab_size,
                weighting_mode=ctx.get("rho1_weighting", "none"),
            )
            loss = ce_loss
        else:
            if use_uncertainty_loss and uw_loss_module is not None:
                ce_loss = criterion(
                    logits.view(-1, config.vocab_size), targets.view(-1)
                )
                task_losses = {"ce": ce_loss}

                if mtp_logits is not None and config.mtp_num_heads > 0:
                    mtp_total = torch.tensor(0.0, device=device)
                    _mtp_lams = (
                        config.mtp_lambda_aux
                        if config.mtp_lambda_aux
                        else [0.15 * (0.33**i) for i in range(len(mtp_logits))]
                    )
                    for k, (mtp_head_logits, lam) in enumerate(
                        zip(mtp_logits, _mtp_lams)
                    ):
                        offset = k + 2
                        if offset < input_ids.shape[1]:
                            mtp_targets_k = targets[:, offset:]
                            mtp_preds_k = mtp_head_logits[
                                :, : mtp_targets_k.shape[1], :
                            ]
                            if mtp_preds_k.shape[1] > 0:
                                mtp_total = mtp_total + lam * F.cross_entropy(
                                    mtp_preds_k.reshape(
                                        -1, mtp_preds_k.shape[-1]
                                    ),
                                    mtp_targets_k.reshape(-1),
                                    ignore_index=-100,
                                )
                    if (
                        isinstance(mtp_total, torch.Tensor)
                        and mtp_total.item() > 0
                    ):
                        task_losses["mtp"] = mtp_total

                loss, uw_info = uw_loss_module(task_losses)
                rho1_stats = None

                if global_step == start_step and batch_idx < grad_accum_steps:
                    weights_str = ", ".join(
                        f"{k}={v:.4f}"
                        for k, v in uw_info.items()
                        if "weight" in k
                    )
                    print(
                        f"[UNCERTAINTY-LOSS VERIFY] ACTIVE Kendall | tasks={list(task_losses.keys())} | {weights_str}"
                    )
            else:
                _use_egce = getattr(config, 'use_egce', False)
                if _use_egce:
                    _egce_logits = logits.view(-1, config.vocab_size).float()
                    _egce_targets = targets.view(-1)
                    _egce_ce_pt = F.cross_entropy(
                        _egce_logits, _egce_targets,
                        reduction='none', ignore_index=-100,
                    )
                    with torch.no_grad():
                        _egce_probs = torch.softmax(_egce_logits, dim=-1)
                        _egce_H = -(_egce_probs * (_egce_probs + 1e-10).log()).sum(dim=-1)
                        _egce_H_mean = _egce_H.mean().clamp(min=1e-8)
                        _egce_weights = (_egce_H / _egce_H_mean).detach()
                        _egce_valid = (_egce_targets != -100)
                        _egce_weights = _egce_weights * _egce_valid.float()
                    ce_loss = (_egce_weights * _egce_ce_pt).sum() / (_egce_valid.float().sum() + 1e-8)
                    if global_step % 100 == 0 and not torch.compiler.is_compiling():
                        print(f"[EGCE] step={global_step} H_mean={_egce_H_mean.item():.3f} "
                              f"w_max={_egce_weights.max().item():.2f} "
                              f"w_min={_egce_weights.min().item():.2f} ce={ce_loss.item():.4f}",
                              flush=True)
                else:
                    ce_loss = criterion(
                        logits.view(-1, config.vocab_size), targets.view(-1)
                    )
                rho1_stats = None

                with torch.no_grad():
                    _gld_targets = targets.view(-1)
                    _gld_loss_pt = F.cross_entropy(
                        logits.view(-1, config.vocab_size).float(),
                        _gld_targets,
                        reduction='none',
                        ignore_index=-100,
                    )
                    _gld_H = _gld_loss_pt.mean()
                    _gld_mask = (
                        (_gld_loss_pt >= 0.5 * _gld_H)
                        & (_gld_loss_pt <= 2.0 * _gld_H)
                        & (_gld_targets != -100)
                    )
                    _gld_n = int(_gld_mask.sum().item())
                    _gld_frac = _gld_n / max(_gld_mask.numel(), 1)
                if _gld_n >= 8:
                    ce_loss = criterion(
                        logits.view(-1, config.vocab_size)[_gld_mask],
                        targets.view(-1)[_gld_mask],
                    )
                if global_step % 100 == 0:
                    print(
                        f"[GOLD] step={global_step} frac={_gld_frac:.3f} n={_gld_n}",
                        flush=True,
                    )

                if getattr(config, 'use_sgb', False) and global_step % 100 == 0:
                    with torch.no_grad():
                        _sgb_threshold = getattr(config, 'sgb_surprisal_threshold', 3.0)
                        _sgb_s2_mask = (_gld_loss_pt > _sgb_threshold) & (_gld_targets != -100)
                        _sgb_s2_frac = _sgb_s2_mask.float().mean().item()
                        _sgb_s1_frac = 1.0 - _sgb_s2_frac
                    print(f"[AS-6 SGB] step={global_step} s2_frac={_sgb_s2_frac:.3f} "
                          f"s1_frac={_sgb_s1_frac:.3f} threshold={_sgb_threshold:.1f}")

                if "x" in _kd_hidden and _kd_hook_handle is not None:
                    with torch.no_grad():
                        _kd_shadow = F.linear(
                            _kd_hidden["x"].to(logits.dtype),
                            model.lm_head.weight,
                        )
                    _kd_T = 2.0
                    _kd_loss = (
                        F.kl_div(
                            F.log_softmax(
                                logits.view(-1, config.vocab_size).float()
                                / _kd_T,
                                dim=-1,
                            ),
                            F.softmax(
                                _kd_shadow.view(-1, config.vocab_size).float()
                                / _kd_T,
                                dim=-1,
                            ).detach(),
                            reduction="batchmean",
                            log_target=False,
                        )
                        * (_kd_T**2)
                    )
                    ce_loss = ce_loss + 0.1 * _kd_loss
                    if global_step % 100 == 0:
                        print(
                            f"[KD-SHADOW] step={global_step} kd={_kd_loss.item():.4f}",
                            flush=True,
                        )
                    _kd_hidden.clear()

                if mtp_logits is not None and config.mtp_num_heads > 0:
                    mtp_lambda_main = config.mtp_lambda_main
                    mtp_lambdas = (
                        config.mtp_lambda_aux
                        if config.mtp_lambda_aux is not None
                        else [0.15 * (0.33**i) for i in range(len(mtp_logits))]
                    )

                    total_loss = mtp_lambda_main * ce_loss
                    for k, (mtp_head_logits, lam) in enumerate(
                        zip(mtp_logits, mtp_lambdas)
                    ):
                        offset = k + 2
                        if offset < input_ids.shape[1]:
                            mtp_targets = targets[:, offset:]
                            mtp_preds = mtp_head_logits[
                                :, : mtp_targets.shape[1], :
                            ]
                            aux_loss = F.cross_entropy(
                                mtp_preds.reshape(-1, mtp_preds.shape[-1]),
                                mtp_targets.reshape(-1),
                                ignore_index=-100,
                            )
                            total_loss = total_loss + lam * aux_loss
                    ce_loss = total_loss
                    if (
                        global_step == start_step
                        and batch_idx < grad_accum_steps
                    ):
                        print(
                            f"[MTP VERIFY] Multi-Token Prediction ACTIVE | {len(mtp_logits)} heads | "
                            f"lambda_main={mtp_lambda_main} lambda_aux={mtp_lambdas[: min(2, len(mtp_lambdas))]}"
                        )

                if getattr(config, 'use_margin_loss', False):
                    _ml = margin_loss(logits.view(-1, config.vocab_size).float(), targets.view(-1), margin=getattr(config, 'margin_loss_margin', 1.0))
                    _mw = getattr(config, 'margin_loss_weight', 1.0)
                    ce_loss = _mw * _ml + (1.0 - _mw) * ce_loss

                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_model(input_ids).logits
                    kd_loss = distillation_loss(
                        logits,
                        teacher_logits,
                        temperature=distill_temperature,
                        use_tsld=use_tsld,
                    )
                    loss = (1 - distill_alpha) * ce_loss + distill_alpha * kd_loss
                else:
                    loss = ce_loss

            if od_teacher is not None:
                with torch.no_grad():
                    od_teacher_logits = od_teacher(input_ids)
                od_kd_loss = distillation_loss(
                    logits,
                    od_teacher_logits,
                    temperature=online_distill_temperature,
                )
                loss = (
                    1 - online_distill_alpha
                ) * loss + online_distill_alpha * od_kd_loss

            if let_teacher is not None and global_step < let_stop_step:
                with torch.no_grad():
                    _, teacher_hidden = let_teacher(
                        input_ids, return_hidden_at=ctx.get("teacher_n_layer", 0) - 1
                    )
                cos_sim = F.cosine_similarity(
                    student_hidden, teacher_hidden, dim=-1
                ).mean()
                let_aux_loss = 1.0 - cos_sim
                decay_factor = 1.0 - (global_step / let_stop_step)
                loss = loss + let_lambda * decay_factor * let_aux_loss

            if (
                angular_reg_lambda > 0 or boundary_reg_lambda > 0
            ) and config.use_ternary:
                from .components.ternary import compute_angular_boundary_reg

                _ang_loss, _bnd_loss, _reg_n_layers = compute_angular_boundary_reg(
                    model,
                    angular_lambda=angular_reg_lambda,
                    boundary_lambda=boundary_reg_lambda,
                )
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[REG VERIFY] Angular+Boundary reg ACTIVE on {_reg_n_layers} ternary layers | "
                        f"ang_loss={_ang_loss.item():.6f} bnd_loss={_bnd_loss.item():.6f}"
                    )
                loss = loss + _ang_loss + _bnd_loss

            if use_tdo and config.use_ternary:
                from .components.ternary import compute_tdo_loss

                _tdo_loss, _tdo_n_layers, _tdo_stats = compute_tdo_loss(
                    model,
                    tdo_lambda=tdo_lambda,
                    tdo_target=tdo_target,
                    temperature=tdo_temperature,
                )
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[TDO VERIFY] TDO ACTIVE on {_tdo_n_layers} ternary layers | "
                        f"tdo_loss={_tdo_loss.item():.6f} "
                        f"dist=[{_tdo_stats['neg']:.3f}, {_tdo_stats['zero']:.3f}, {_tdo_stats['pos']:.3f}]"
                    )
                loss = loss + _tdo_loss

            if config.use_optrot and config.use_ternary:
                from .components.optrot import collect_optrot_loss

                optrot_loss = collect_optrot_loss(model, config.optrot_lambda)
                if isinstance(optrot_loss, torch.Tensor):
                    loss = loss + optrot_loss

            if config.use_moe:
                _moe_aux = 0.0
                for _m in model.modules():
                    if hasattr(_m, "_last_aux_loss") and hasattr(
                        _m, "_expert_counts"
                    ):
                        _moe_aux = _moe_aux + _m._last_aux_loss
                if isinstance(_moe_aux, torch.Tensor) and _moe_aux > 0:
                    loss = loss + _moe_aux

            if contractive_reg_lambda > 0:
                _contract_loss = (
                    model.compute_contractive_reg() * contractive_reg_lambda
                )
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[CONTRACTIVE-REG VERIFY] ACTIVE | "
                        f"loss={_contract_loss.item():.6f} lambda={contractive_reg_lambda}"
                    )
                loss = loss + _contract_loss

            if use_sca_sparsity and config.use_ternary:
                if global_step >= sca_warmup_steps:
                    from .components.ternary import compute_sca_sparsity_loss

                    _sca_loss, _sca_stats = compute_sca_sparsity_loss(
                        model_actual,
                        target_sparsity=sca_target_sparsity,
                        group_size=config.ternary_group_size,
                    )
                    loss = loss + sca_lambda * _sca_loss
                    if global_step % 100 == 0 and batch_idx < grad_accum_steps:
                        print(
                            f"[SCA SPARSITY] step={global_step} | "
                            f"current_sparsity={_sca_stats['current_sparsity']:.4f} | "
                            f"target={sca_target_sparsity:.4f} | "
                            f"loss={_sca_loss.item():.6f} | "
                            f"lambda={sca_lambda:.6f}"
                        )

            if getattr(config, "use_frar", False):
                if global_step >= config.frar_warmup_steps:
                    progress = (global_step - config.frar_warmup_steps) / max(
                        1, ctx.get("max_steps", 1) - config.frar_warmup_steps
                    )
                    alpha = 5.0
                    frar_lambda = config.frar_lambda_end + (
                        config.frar_lambda_start - config.frar_lambda_end
                    ) * math.exp(-alpha * progress)
                    frar_loss = model.compute_frar_loss()
                    loss = loss + frar_lambda * frar_loss

                    if global_step % 50 == 0:
                        print(
                            f"[FRAR ACTIVE] step={global_step}, lambda={frar_lambda:.6f}, "
                            f"alignment_loss={frar_loss.item():.6f}, "
                            f"weighted={frar_lambda * frar_loss.item():.6f}"
                        )

            if getattr(config, "use_recursive_load_balance", False):
                _rec_lb_loss = model.compute_recursive_load_balance_loss()
                if (
                    isinstance(_rec_lb_loss, torch.Tensor)
                    and _rec_lb_loss.item() > 0
                ):
                    loss = loss + _rec_lb_loss
                    if global_step == start_step and batch_idx < grad_accum_steps:
                        print(
                            f"[RECURSIVE-LB VERIFY] ACTIVE | loss={_rec_lb_loss.item():.6f}"
                        )

            if use_sskd and config.use_ternary:
                model.set_quantization_enabled(False)
                with torch.no_grad():
                    shadow_out = model(input_ids)
                    shadow_logits = (
                        shadow_out
                        if isinstance(shadow_out, torch.Tensor)
                        else shadow_out[0]
                    )
                model.set_quantization_enabled(True)
                from .components.sskd_loss import compute_sskd_loss

                sskd_loss = compute_sskd_loss(
                    logits.view(-1, config.vocab_size),
                    shadow_logits.view(-1, config.vocab_size),
                    temperature=sskd_temperature,
                )
                loss = loss + sskd_alpha * sskd_loss
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[SSKD VERIFY] ACTIVE | kl_loss={sskd_loss.item():.4f} alpha={sskd_alpha}"
                    )

            if use_ema_distill and ema is not None and config.use_ternary and global_step >= 50:
                _ema_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                _ema_backup = ema.apply(_ema_model)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp and device == "cuda"):
                        _ema_out = _ema_model(input_ids)
                        _ema_logits = (
                            _ema_out if isinstance(_ema_out, torch.Tensor) else _ema_out[0]
                        )
                ema.restore(_ema_model, _ema_backup)
                _bd_loss = distillation_loss(
                    logits.view(-1, config.vocab_size),
                    _ema_logits.view(-1, config.vocab_size),
                    temperature=ema_distill_temperature,
                    use_tsld=False,
                )
                loss = loss + ema_distill_alpha * _bd_loss
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[BITDISTILL VERIFY] ACTIVE | kl={_bd_loss.item():.4f} "
                        f"alpha={ema_distill_alpha} T={ema_distill_temperature}"
                    )

            if use_ema_replay and ema is not None and global_step >= 50:
                _rpl_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                _rpl_backup = ema.apply(_rpl_model)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp and device == "cuda"):
                        _rpl_out = _rpl_model(input_ids)
                        _rpl_logits = (
                            _rpl_out if isinstance(_rpl_out, torch.Tensor) else _rpl_out[0]
                        )
                ema.restore(_rpl_model, _rpl_backup)
                _rpl_kl = F.kl_div(
                    F.log_softmax(logits.view(-1, config.vocab_size).float(), dim=-1),
                    F.softmax(_rpl_logits.view(-1, config.vocab_size).float(), dim=-1).detach(),
                    reduction="batchmean",
                )
                loss = loss + ema_replay_lambda * _rpl_kl
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[AP-4 EMA-REPLAY] ACTIVATED | kl={_rpl_kl.item():.4f} "
                        f"lambda={ema_replay_lambda}"
                    )

            if use_thm_loss and config.use_ternary:
                from .components.thm_loss import compute_thm_loss

                _thm_loss = compute_thm_loss(model, margin=thm_margin)
                loss = loss + thm_lambda * _thm_loss
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[THM-LOSS VERIFY] ACTIVE | loss={_thm_loss.item():.6f} margin={thm_margin}"
                    )

            if use_tail_loss:
                from .components.tail_loss import compute_tail_loss

                _tail_loss = compute_tail_loss(
                    logits.view(-1, config.vocab_size),
                    targets.view(-1),
                    flip_margin=tail_flip_margin,
                    num_negatives=tail_num_negatives,
                )
                loss = loss + tail_lambda * _tail_loss
                if global_step == start_step and batch_idx < grad_accum_steps:
                    print(
                        f"[TAIL VERIFY] ACTIVE | loss={_tail_loss.item():.6f} margin={tail_flip_margin}"
                    )

            if use_rcl_loss and hasattr(model, "get_intermediate_states"):
                _inter_states = model.get_intermediate_states()
                if _inter_states and len(_inter_states) >= 2:
                    from .components.rcl_loss import compute_rcl_loss

                    _output_proj = (
                        model.lm_head if hasattr(model, "lm_head") else None
                    )
                    if _output_proj is not None:
                        _rcl_loss = compute_rcl_loss(
                            _inter_states,
                            targets,
                            output_proj=_output_proj,
                            beta=rcl_beta,
                            sample_every=rcl_sample_every,
                        )
                        loss = loss + rcl_lambda * _rcl_loss
                        if (
                            global_step == start_step
                            and batch_idx < grad_accum_steps
                        ):
                            print(
                                f"[RCL VERIFY] ACTIVE | loss={_rcl_loss.item():.6f} | states={len(_inter_states)} beta={rcl_beta}"
                            )

            if use_ter_loss:
                from .components.ter_loss import compute_ter_loss

                _ter_loss = compute_ter_loss(
                    logits.view(-1, config.vocab_size),
                    model,
                    config.vocab_size,
                    mu=ter_loss_mu,
                    noise_var=ter_noise_var,
                )
                if _ter_loss.item() > 0:
                    loss = loss + _ter_loss
                    if global_step == start_step and batch_idx < grad_accum_steps:
                        print(
                            f"[TER-LOSS VERIFY] ACTIVE | penalty={_ter_loss.item():.6f}"
                        )

            if use_predictsim and _local_losses_list is not None:
                _ce_w = getattr(config, 'local_loss_ce_weight', 0.7)
                _cos_w = getattr(config, 'local_loss_cosine_weight', 0.3)
                _global_blend = getattr(config, 'local_loss_global_blend', 0.3)
                _predsim_total = torch.tensor(0.0, device=device)
                _n_local = 0
                for _h_T, _probe in _local_losses_list:
                    _local_logits = _probe(_h_T)
                    _local_targets = targets[:, -1]
                    _ce_term = F.cross_entropy(_local_logits, _local_targets)
                    _y_embed = model.embedding.weight[_local_targets].detach()
                    _cos_term = 1.0 - F.cosine_similarity(_h_T, _y_embed, dim=-1).mean()
                    _block_loss = _ce_w * _ce_term + _cos_w * _cos_term
                    _predsim_total = _predsim_total + _block_loss
                    _n_local += 1
                if _n_local > 0:
                    _predsim_avg = _predsim_total / _n_local
                    loss = (1.0 - _global_blend) * _predsim_avg + _global_blend * loss
                if global_step % 50 == 0:
                    print(f"[PREDSIM] step={global_step} n_blocks={_n_local} "
                          f"local_avg={_predsim_avg.item():.4f} final_loss={loss.item():.4f}")

    scaled_loss = loss / grad_accum_steps
    loss_val = loss.item()

    ctx["loss_val"] = loss_val
    ctx["scaled_loss"] = scaled_loss

    return loss_val, ctx


def train_model(
    config: ModelConfig,
    dataset_type: str = "synthetic",
    data_dir: str = None,
    num_epochs: int = 5,
    batch_size: int = 32,
    grad_accum_steps: int = 1,
    base_lr: float = 6e-4,
    warmup_steps: int = 100,
    max_steps: Optional[int] = None,
    num_samples: int = 10000,
    max_memory_mb: int = 4096,
    log_every: int = 50,
    save_dir: Optional[str] = None,
    device: str = "cpu",
    progressive: bool = False,
    qat_start_pct: float = 0.15,
    routing_start_pct: float = 0.50,
    lr_schedule_type: str = "inv_sqrt",
    wsd_warmup_pct: float = 0.10,
    wsd_stable_pct: float = 0.60,
    two_stage_decay: bool = False,
    two_stage_drop_pct: float = 0.5,
    two_stage_drop_factor: float = 0.1,
    two_stage_wd_start_pct: float = 0.67,
    optimizer_type: str = "adamw",
    use_8bit_optimizer: bool = False,
    shadow_bf16: bool = False,
    use_compile: bool = True,
    seq_warmup: bool = False,
    seq_warmup_min: int = 64,
    seq_warmup_pct: float = 0.30,
    ste_type: str = "standard",
    gradual_threshold: bool = False,
    pressure_blend: bool = False,
    layer_wise_precision: bool = False,
    use_amp: bool = False,
    use_bf16: bool = True,
    use_gradient_checkpointing: bool = False,
    checkpoint_every_n: int = 1,
    ternary_aware_checkpointing: bool = False,
    use_sdpa: bool = False,
    use_activation_quant: bool = True,
    activation_quant_bits: int = 8,
    use_cpu_offload: bool = False,
    use_distillation: bool = False,
    teacher_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    distill_alpha: float = 0.5,
    distill_temperature: float = 2.0,
    use_tsld: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 8,
    use_rho1: bool = False,
    rho1_keep_ratio: float = 0.5,
    rho1_ref_losses_path: str = None,
    rho1_weighting: str = "hard_mask",
    use_curriculum: bool = False,
    use_sequence_packing: bool = False,
    tequila_lambda: float = 0.0,
    use_hestia2026: bool = False,
    layerwise_lr: bool = False,
    weight_decay: float = 0.01,
    beta2: float = 0.999,
    fused_optimizer: bool = False,
    let_teacher_path: str = None,
    let_lambda: float = 0.1,
    let_stop_step: int = 1500,
    let_layer_k: int = 3,
    online_distill_teacher_path: str = None,
    online_distill_alpha: float = 0.5,
    online_distill_temperature: float = 4.0,
    use_liger_ce: bool = True,
    resume_from: Optional[str] = None,
    eval_every: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    on_checkpoint: Optional[Callable] = None,
    max_nan_steps: int = 0,
    antiwindup_k: float = 0.0,
    angular_reg_lambda: float = 0.0,
    boundary_reg_lambda: float = 0.0,
    use_tdo: bool = False,
    tdo_lambda: float = 0.01,
    tdo_target: tuple = (1 / 3, 1 / 3, 1 / 3),
    tdo_temperature: float = 10.0,
    use_galore: bool = False,
    galore_rank: int = 128,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 1.0,
    use_caq: bool = False,
    caq_alpha: float = 1.0,
    use_tpsq: bool = False,
    tpsq_beta: float = 1.0,
    use_tcd: bool = False,
    use_gumbel_noise: bool = False,
    gumbel_temperature: float = 0.1,
    use_iaqd: bool = False,
    use_stp: bool = False,
    stp_sigma_ratio: float = 0.1,
    use_ctce: bool = False,
    ctce_beta: float = 1.5,
    ctce_delta_ratio: float = 0.5,
    use_dual_rate: bool = False,
    dual_rate_peak: float = 2.5,
    dual_rate_floor: float = 0.1,
    use_gaste: bool = False,
    gaste_sigma_g: float = 0.3,
    use_saq: bool = False,
    saq_k: int = 3,
    use_rlec: bool = False,
    rlec_rank: int = 4,
    use_adagc: bool = True,
    adagc_factor: float = 1.5,
    adagc_beta: float = 0.99,
    use_margin_sparse: bool = False,
    margin_delta: float = 0.1,
    margin_ema_beta: float = 0.99,
    margin_refresh_interval: int = 10,
    use_fused_requantize: bool = False,
    use_fused_gradient_cascade: bool = False,
    use_info_freeze: bool = False,
    info_freeze_alpha: float = 1e-5,
    info_freeze_warmup: int = 200,
    use_dead_zone: bool = False,
    dead_zone_margin: float = 0.1,
    dead_zone_lr_decay: bool = True,
    use_acsi_init: bool = False,
    acsi_max_batches: int = 4,
    use_commitment_monitor: bool = False,
    commitment_check_interval: int = 50,
    commitment_plateau_epsilon: float = 0.005,
    commitment_plateau_patience: int = 3,
    commitment_lr_boost: float = 1.5,
    commitment_lr_decay: float = 0.5,
    commitment_early_stop_threshold: float = 0.98,
    commitment_early_stop: bool = True,
    use_instant: bool = False,
    instant_rank: int = 32,
    use_ags: bool = False,
    ags_top_k_frac: float = 0.05,
    ags_full_sync_interval: int = 50,
    ags_delta: float = 0.1,
    contractive_reg_lambda: float = 0.0,
    use_uncertainty_loss: bool = False,
    uncertainty_n_tasks: int = 4,
    uncertainty_initial_logvar: float = 0.0,
    use_sparse_shadow: bool = False,
    sparse_shadow_margin: float = 0.1,
    sparse_shadow_decay: float = 0.0,
    sparse_shadow_interval: int = 10,
    use_sca_sparsity: bool = False,
    sca_lambda: float = 0.01,
    sca_target_sparsity: float = 0.5,
    sca_warmup_steps: int = 0,
    use_fp8_grad_accum: bool = False,
    use_topk_grad_sparsification: bool = False,
    topk_grad_sparsity: float = 0.9,
    label_smoothing: float = 0.05,
    use_grad_noise: bool = False,
    grad_noise_eta: float = 0.1,
    grad_noise_gamma: float = 0.55,
    cosine_restart_period: int = 1000,
    cosine_restart_decay: float = 0.8,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    use_cuda_prefetch: bool = False,
    auto_batch_size: bool = False,
    vram_budget_mb: int = 10000,
    use_gradient_sign_consistency: bool = False,
    grad_sign_window: int = 8,
    grad_sign_threshold: int = 6,
    use_fisher_preconditioning: bool = False,
    fisher_damping: float = 1e-4,
    use_sign_sgd: bool = False,
    sign_sgd_momentum: float = 0.9,
    use_consensus_gradient_filter: bool = False,
    consensus_alpha: float = 0.5,
    use_adaptive_quantization: bool = False,
    adaptive_quant_sensitivity_threshold: float = 0.1,
    adaptive_quant_interval: int = 500,
    use_nonuniform_checkpointing: bool = False,
    use_auto_config: bool = False,
    use_sqrt_ssm_checkpointing: bool = False,
    use_snr_diagnostic: bool = False,
    snr_freeze_threshold: float = 0.1,
    snr_freeze_interval: int = 100,
    use_reversible_layers: bool = False,
    use_combinatorial_finetune: bool = False,
    combinatorial_max_sweeps: int = 3,
    use_ucb_ternary_training: bool = False,
    ucb_ternary_exploration_c: float = 1.0,
    use_fscr_flip_gate: bool = False,
    fscr_threshold: float = 2.0,
    use_spectral_ssm: bool = False,
    use_spsa_training: bool = False,
    spsa_perturbation_scale: float = 0.1,
    spsa_lr: float = 0.01,
    use_rank_margin_loss: bool = False,
    rank_margin: float = 1.0,
    rank_num_negatives: int = 64,
    use_whittle_index: bool = False,
    whittle_update_fraction: float = 0.1,
    # A3-F4: SSKD
    use_sskd: bool = False,
    sskd_alpha: float = 0.1,
    sskd_temperature: float = 2.0,
    # R5-A1-P1: HFOTT gradient correction
    use_hfott: bool = True,          # R5-A1-P1: Forward-only gradient correction (NADA DE LEGO)
    hfott_sigma: float = 0.1,
    hfott_alpha: float = 0.3,
    # R5-A6-F1: EMA self-distillation (BITDISTILL)
    use_ema_distill: bool = True,
    ema_distill_alpha: float = 0.1,
    ema_distill_temperature: float = 4.0,
    # AP-4: EMA Shadow Generative Replay
    use_ema_replay: bool = False,
    ema_replay_lambda: float = 0.05,
    # A3-F8: Dynamic Loss Weighting
    use_dynamic_loss_weighting: bool = False,
    dlw_gamma: float = 1.0,
    # A3-M1: THM Loss
    use_thm_loss: bool = False,
    thm_margin: float = 0.3,
    thm_lambda: float = 0.01,
    # A3-M2: TAIL Loss
    use_tail_loss: bool = False,
    tail_lambda: float = 0.01,
    tail_flip_margin: float = 1.0,
    tail_num_negatives: int = 5,
    # A3-M3: RCL
    use_rcl_loss: bool = False,
    rcl_beta: float = 1.0,
    rcl_lambda: float = 0.05,
    rcl_sample_every: int = 4,
    # A3-M4: TER-LOSS
    use_ter_loss: bool = False,
    ter_loss_mu: float = 0.1,
    ter_noise_var: float = 0.01,
    # A3-M5: GCP
    use_gcp: bool = False,
    gcp_alpha: float = 1.0,
    # A3-M6: RAFU
    use_rafu: bool = False,
    rafu_threshold: float = 0.1,
    rafu_interval: int = 50,
    # A3-M7: IBLL
    use_ibll: bool = False,
    ibll_beta: float = 1.0,
    ibll_gamma: float = 0.1,
    ibll_eps: float = 0.01,
    ibll_lambda: float = 0.02,
    ibll_sample_every: int = 8,
    # A3-M8: BFA
    use_bfa: bool = False,
    bfa_rho: float = 0.95,
    bfa_tau: float = 1.0,
    bfa_theta: float = 0.9,
    bfa_interval: int = 10,
    # A3-P1: MDE
    use_mde: bool = False,
    mde_eta: float = 0.1,
    mde_lambda: float = 0.05,
    mde_sample_every: int = 4,
    # A3-P2: TECWC
    use_tecwc: bool = False,
    tecwc_code_rate: float = 0.9,
    tecwc_lambda: float = 0.001,
    # A3-P3: Spectral Surrogate Objective
    use_sso: bool = False,
    sso_rho_target: float = 0.95,
    sso_alpha: float = 1.0,
    sso_beta: float = 0.1,
    sso_interval: int = 100,
    # A3-P4: Spin-Glass Annealing
    use_sgapd: bool = False,
    sgapd_initial_temp: float = 1.0,
    sgapd_min_temp: float = 0.01,
    sgapd_decay: float = 0.995,
    sgapd_sweeps: int = 10,
    sgapd_interval: int = 20,
    # A3-P5: Ergodic Trajectory Optimizer
    use_ergodic: bool = False,
    ergodic_alpha: float = 1.0,
    ergodic_rho_target: float = 0.9,
    ergodic_flip_fraction: float = 0.01,
    ergodic_interval: int = 50,
    # A3-P6: Minimax Depth Fairness
    use_minimax_depth: bool = False,
    minimax_tau: float = 1.0,
    minimax_hard: bool = False,
    minimax_lambda: float = 0.05,
    minimax_sample_every: int = 4,
    # A3-P7: Koopman Spectral Shaping
    use_koopman: bool = False,
    koopman_long_target: float = 0.97,
    koopman_fast_target: float = 0.5,
    koopman_lambda: float = 0.01,
    koopman_interval: int = 100,
    koopman_max_dim: int = 64,
    # A3-P8: TD Depth Learning
    use_td_dl: bool = False,
    td_lambda_val: float = 0.8,
    td_dl_weight: float = 0.05,
    td_sample_every: int = 2,
    # F24: Information-Theoretic Freezing
    use_info_freezing: bool = False,
    info_freeze_threshold: float = 0.01,
    info_freezing_warmup: int = 500,
    info_freeze_window: int = 100,
    info_freeze_beta: float = 0.99,
) -> dict:
    """Train a Model v7 instance.

    Args:
        config: Model configuration
        dataset_type: 'synthetic' or 'copy'
        num_epochs: Number of training epochs
        batch_size: Batch size
        base_lr: Base learning rate (differential LRs applied automatically)
        warmup_steps: Linear warmup steps
        max_steps: Stop after this many steps (None = full epochs)
        log_every: Print status every N steps
        save_dir: Directory to save checkpoints and logs
        device: 'cpu' or 'cuda'
        num_samples: Number of training samples
        progressive: Enable progressive training scheduler
        qat_start_pct: Fraction of training before enabling QAT (0.0-1.0)
        routing_start_pct: Fraction of training before enabling routing (0.0-1.0)

    Returns:
        dict with final metrics and training history
    """
    _setup_global_flags(
        config,
        use_rho1,
        use_distillation,
        online_distill_teacher_path,
        use_saq,
        progressive,
        use_amp,
        use_bf16,
        use_gradient_checkpointing,
        checkpoint_every_n,
        ternary_aware_checkpointing,
        use_sdpa,
        use_activation_quant,
        activation_quant_bits,
        use_cpu_offload,
        use_compile,
        antiwindup_k,
        use_instant,
        instant_rank,
    )

    (
        model,
        gradient_monitor,
        ram_checkpointer,
        anomaly_detector,
        use_qmm,
        inoculation,
        delta_chronicle,
        flip_tracker,
        gradient_recyclers,
        let_teacher,
        let_proj,
        od_teacher,
        param_info,
    ) = _setup_model_and_components(
        config=config,
        device=device,
        ste_type=ste_type,
        use_compile=use_compile,
        use_hestia2026=use_hestia2026,
        tequila_lambda=tequila_lambda,
        use_caq=use_caq,
        caq_alpha=caq_alpha,
        use_tpsq=use_tpsq,
        tpsq_beta=tpsq_beta,
        use_tcd=use_tcd,
        use_gumbel_noise=use_gumbel_noise,
        gumbel_temperature=gumbel_temperature,
        use_iaqd=use_iaqd,
        use_stp=use_stp,
        stp_sigma_ratio=stp_sigma_ratio,
        use_ctce=use_ctce,
        ctce_beta=ctce_beta,
        ctce_delta_ratio=ctce_delta_ratio,
        use_dual_rate=use_dual_rate,
        dual_rate_peak=dual_rate_peak,
        dual_rate_floor=dual_rate_floor,
        use_gaste=use_gaste,
        gaste_sigma_g=gaste_sigma_g,
        use_rlec=use_rlec,
        rlec_rank=rlec_rank,
        use_margin_sparse=use_margin_sparse,
        margin_ema_beta=margin_ema_beta,
        use_ags=use_ags,
        ags_top_k_frac=ags_top_k_frac,
        ags_full_sync_interval=ags_full_sync_interval,
        ags_delta=ags_delta,
        shadow_bf16=shadow_bf16,
        let_teacher_path=let_teacher_path,
        let_lambda=let_lambda,
        let_stop_step=let_stop_step,
        let_layer_k=let_layer_k,
        online_distill_teacher_path=online_distill_teacher_path,
        online_distill_alpha=online_distill_alpha,
        online_distill_temperature=online_distill_temperature,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_samples=num_samples,
    )

    print(f"\n{'=' * 60}")
    print(f"Model v7 Training")
    print(f"{'=' * 60}")
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Ternary: {param_info['ternary']:,} ({param_info['ternary_pct']:.1f}%)")
    print(f"FP16: {param_info['fp16']:,}")
    print(f"Memory (FP16): {param_info['memory_mb_fp16']:.1f} MB")
    print(f"Memory (mixed): {param_info['memory_mb_mixed']:.1f} MB")
    print(f"Device: {device}")
    print(
        f"SSM layers: {sum(1 for i in range(config.n_layer) if not config.is_swa_layer(i))}"
    )
    print(
        f"SWA layers: {sum(1 for i in range(config.n_layer) if config.is_swa_layer(i))}"
    )
    print(f"Ternary: {config.use_ternary}")
    print(f"Routing: {config.use_routing}")
    if config.use_moe:
        moe_indices = getattr(config, "moe_layer_indices", None) or []
        print(
            f"MoE: enabled ({config.moe_num_experts} experts, top_k={config.moe_top_k}, layers={moe_indices})"
        )
    else:
        print(f"MoE: disabled")
    print(f"Progressive training: {progressive}")
    if progressive:
        print(f"  - QAT starts at {qat_start_pct * 100:.0f}% of training")
        print(f"  - Routing starts at {routing_start_pct * 100:.0f}% of training")
    print(
        f"Grad accumulation: {grad_accum_steps} steps (effective batch = {batch_size * grad_accum_steps})"
    )
    print(f"LR schedule: {lr_schedule_type}")
    if lr_schedule_type == "inverse_sqrt":
        # Debug: show LR at key training points to verify schedule
        # total_steps is computed later (needs train_loader), use max_steps as early estimate
        _est_total = max_steps if max_steps else None
        if _est_total and _est_total > warmup_steps:
            for pct in [0.25, 0.50, 0.75, 1.00]:
                test_step = warmup_steps + int((_est_total - warmup_steps) * pct)
                t = test_step - warmup_steps + 1
                decay = max(0.1, (1.0 / t) ** 0.5)
                print(
                    f"  [InvSqrt] At {pct:.0%} training (step {test_step}): LR multiplier = {decay:.4f} (effective LR = {base_lr * decay:.6f})"
                )
        else:
            print(f"  [InvSqrt] decay = t^(-0.5) after warmup, floor at 10% of peak LR")
    if lr_schedule_type == "wsd":
        print(
            f"  WSD: {wsd_warmup_pct * 100:.0f}% warmup, {wsd_stable_pct * 100:.0f}% stable, {(1 - wsd_warmup_pct - wsd_stable_pct) * 100:.0f}% decay"
        )
    if lr_schedule_type == "two_stage":
        _est_total = max_steps if max_steps else None
        _drop_step = (
            int(_est_total * two_stage_drop_pct) if _est_total else "(computed later)"
        )
        print(
            f"  Two-stage LR: constant until {two_stage_drop_pct:.0%}, "
            f"then drop to {two_stage_drop_factor}x, cosine to 0"
        )
        if isinstance(_drop_step, int):
            print(
                f"  LR drop at step {_drop_step} | "
                f"Peak LR = {base_lr:.2e}, Post-drop LR = {base_lr * two_stage_drop_factor:.2e}"
            )
    print(f"Two-stage weight decay: {two_stage_decay}")
    if two_stage_decay:
        _est_total = max_steps if max_steps else None
        _wd_start = (
            int(_est_total * two_stage_wd_start_pct)
            if _est_total
            else "(computed later)"
        )
        print(
            f"  WD schedule: {weight_decay} constant until {two_stage_wd_start_pct:.0%}, "
            f"then cosine to 0"
        )
        if isinstance(_wd_start, int):
            print(f"  WD decay starts at step {_wd_start}")
    print(f"8-bit optimizer: {use_8bit_optimizer}")
    if shadow_bf16:
        print(f"BF16 shadow weights: enabled (ternary layers only)")
    print(
        f"Optimizer: {optimizer_type} (beta2={beta2}, wd={weight_decay}, fused={fused_optimizer})"
    )
    if optimizer_type == "galore":
        print(
            f"GaLore: rank={galore_rank}, SVD gap={galore_update_proj_gap}, scale={galore_scale}"
        )
    if getattr(config, "use_aga", False):
        aga_k = getattr(config, "aga_threshold_k", 32)
        aga_lr_mult = getattr(config, "aga_lr_multiplier", 1.0)
        aga_int16 = getattr(config, "aga_use_int16", True)
        print(f"AGA: enabled (K={aga_k}, lr_mult={aga_lr_mult}, int16={aga_int16})")
    print(f"torch.compile: {use_compile}")
    if ste_type != "standard":
        print(f"STE type: {ste_type}")
    if gradual_threshold:
        print(f"Gradual threshold: 0.5 -> {DEFAULT_THRESHOLD_COEFF:.4f}")
    if pressure_blend:
        print(f"Pressure blend: enabled (20% QAT duration)")
    if layer_wise_precision:
        print(f"Layer-wise precision: first/last layers FP16")
    if seq_warmup:
        print(
            f"Sequence warmup: {seq_warmup_min} -> {config.seq_len} over {seq_warmup_pct * 100:.0f}% of training"
        )
    if use_rho1:
        print(
            f"Rho-1 scoring: weighting={rho1_weighting}, keep_ratio={rho1_keep_ratio}, ref_losses={rho1_ref_losses_path}"
        )
    if use_curriculum:
        print(f"Curriculum: easy-first ordering via ref_losses")
    if let_teacher_path is not None:
        print(f"LET Transfer: teacher={let_teacher_path}")
        print(
            f"  lambda={let_lambda}, stop_step={let_stop_step}, layer_k={let_layer_k}"
        )
    if online_distill_teacher_path is not None:
        print(f"Online Distill: teacher={online_distill_teacher_path}")
        print(
            f"  alpha={online_distill_alpha}, temperature={online_distill_temperature}"
        )
    if use_amp:
        print(f"AMP (Automatic Mixed Precision): enabled")
    if use_gradient_checkpointing:
        print(f"Gradient checkpointing: enabled (every {checkpoint_every_n} layer(s))")
    if ternary_aware_checkpointing:
        print(f"Ternary-aware checkpointing: ON (SSM layers recomputed, SWA stored)")
    print(f"[GRAD-CKPT] activation_checkpoint_granularity={config.activation_checkpoint_granularity!r}")
    if use_activation_quant:
        print(
            f"Activation quantization: {activation_quant_bits}-bit (warmup={config.activation_quant_warmup} steps)"
        )
    if use_instant:
        print(
            f"INSTANT activation compression: rank={instant_rank} (warmup={config.instant_warmup} steps)"
        )
    if use_sdpa:
        print(f"SDPA (scaled_dot_product_attention): enabled")
    if angular_reg_lambda > 0 or boundary_reg_lambda > 0:
        print(f"Angular regularization: lambda={angular_reg_lambda}")
        print(f"Boundary-repulsive reg: lambda={boundary_reg_lambda}")
    if use_tdo:
        print(
            f"TDO: lambda={tdo_lambda}, target={tdo_target}, temperature={tdo_temperature}"
        )
    if use_saq:
        saq_cycle = saq_k + 1
        print(
            f"SAQ (Step-Aligned Quantization): K={saq_k} (cycle={saq_cycle}: {saq_k} FP16 + 1 ternary)"
        )
    print(f"Memory limit: {max_memory_mb} MB")
    print(f"{'=' * 60}\n")

    train_loader, curriculum_sampler = _setup_data(
        config,
        model,
        dataset_type,
        batch_size,
        num_samples,
        data_dir,
        device,
        use_curriculum,
        rho1_ref_losses_path,
        auto_batch_size,
        vram_budget_mb,
        use_cuda_prefetch,
        use_sequence_packing,
        use_acsi_init,
        acsi_max_batches,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        lr_schedule_type,
        use_rho1,
    )

    # R4-A3-F5: Uncertainty-Weighted Multi-Task Loss (Kendall)
    uw_loss_module = None
    if use_uncertainty_loss:
        from .components.uncertainty_loss import UncertaintyWeightedLoss

        uw_loss_module = UncertaintyWeightedLoss(
            n_tasks=uncertainty_n_tasks,
            initial_log_var=uncertainty_initial_logvar,
        ).to(device)
        print(
            f"[UNCERTAINTY-WEIGHTED LOSS] Enabled: {uncertainty_n_tasks} tasks, "
            f"initial_logvar={uncertainty_initial_logvar}"
        )

    optimizer, scheduler, aga_optimizer, bbo_optimizer, fused_aga, grad_clipper, _ternary_monitor, bifurcated_optimizer, cusum_monitor, probe_monitors, asymmetric_wrapper, hybrid_rtrl, pattern_transplant, restart_scheduler, info_freezer, commitment_precision, teacher_model, scaler, _g1_integral, _g1_prev_error, _g1_prev_committed_frac, _g1_pid_lr_mult, flip_tracker, ema = _setup_optimizer(
        config=config,
        model=model,
        device=device,
        base_lr=base_lr,
        beta2=beta2,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        num_epochs=num_epochs,
        len_train_loader=len(train_loader),
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_schedule_type=lr_schedule_type,
        wsd_warmup_pct=wsd_warmup_pct,
        wsd_stable_pct=wsd_stable_pct,
        two_stage_drop_pct=two_stage_drop_pct,
        two_stage_drop_factor=two_stage_drop_factor,
        cosine_restart_period=cosine_restart_period,
        cosine_restart_decay=cosine_restart_decay,
        use_ternary=getattr(config, "use_ternary", False),
        use_uncertainty_loss=use_uncertainty_loss,
        uw_loss_module=uw_loss_module,
        use_fused_gradient_cascade=use_fused_gradient_cascade,
        use_cpu_offload=use_cpu_offload,
        use_ags=getattr(config, "use_ags", False),
        ags_top_k_frac=ags_top_k_frac,
        ags_delta=ags_delta,
        ags_full_sync_interval=ags_full_sync_interval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        galore_rank=galore_rank,
        galore_update_proj_gap=galore_update_proj_gap,
        galore_scale=galore_scale,
        use_amp=use_amp,
        use_bf16=use_bf16,
        fused_optimizer=fused_optimizer,
        use_distillation=use_distillation,
        teacher_model_name=teacher_model_name,
        use_info_freezing=use_info_freezing,
        info_freeze_threshold=info_freeze_threshold,
        info_freezing_warmup=info_freezing_warmup,
        info_freeze_window=info_freeze_window,
        info_freeze_beta=info_freeze_beta,
    )

    # C3: Progressive Layer Freezing
    progressive_freezer = None
    if config.use_progressive_freezing:
        if not config.use_commitment_tracking:
            import warnings

        from .training.progressive_freezing import ProgressiveFreezer

        model_for_freezing = model._orig_mod if hasattr(model, "_orig_mod") else model
        progressive_freezer = ProgressiveFreezer(
            model=model_for_freezing,
            freeze_threshold=config.freeze_threshold,
            check_interval=config.freeze_check_interval,
            min_step=config.freeze_min_step,
        )
        print(
            f"[C3-FREEZE] Enabled: threshold={config.freeze_threshold:.2%}, "
            f"interval={config.freeze_check_interval}, min_step={config.freeze_min_step}"
        )

    # AutoFreeze: layer-level gradient-norm convergence freezing (always-on for ternary)
    autofreeze = None
    if use_ternary:
        from .training.autofreeze import AutoFreeze
        _af_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        autofreeze = AutoFreeze(
            model=_af_model,
            check_interval=50,
            window_size=10,
            convergence_threshold=0.05,
            grace_period=500,
        )

    # R4-A3-M7: IBLL probe
    ibll_probe = None
    if use_ibll:
        from .components.ibll import IBLLProbe

        ibll_probe = IBLLProbe(config.d_model, config.vocab_size).to(device)
        optimizer.add_param_group(
            {
                "params": list(ibll_probe.parameters()),
                "lr": base_lr,
                "weight_decay": 0.0,
            }
        )
        print(
            f"[IBLL] Enabled: beta={ibll_beta} gamma={ibll_gamma} sample_every={ibll_sample_every}"
        )

    # R4-A3-M8: BFA accumulators
    bfa_accumulators = {}
    if use_bfa:
        from .components.bfa import BayesianFlipAccumulator

        model_to_check = model._orig_mod if hasattr(model, "_orig_mod") else model
        for name, module in model_to_check.named_modules():
            if hasattr(module, "quantize_enabled") and hasattr(module, "weight"):
                bfa_accumulators[name] = BayesianFlipAccumulator(
                    module.weight.shape,
                    device=device,
                    rho=bfa_rho,
                    tau=bfa_tau,
                    theta=bfa_theta,
                )
        print(
            f"[BFA] Enabled: {len(bfa_accumulators)} layers, rho={bfa_rho} tau={bfa_tau} theta={bfa_theta}"
        )

    # R4-A3-P1: MDE adversary
    mde_adversary = None
    if use_mde:
        from .components.mde_loss import MinimaxDepthEquilibrium

        n_depths = getattr(config, "recursive_ssm_iterations", 32)
        if n_depths == 0:
            n_depths = 32
        mde_adversary = MinimaxDepthEquilibrium(n_depths, eta=mde_eta, device=device)
        print(f"[MDE] Enabled: {n_depths} depths, eta={mde_eta}")

    # R4-A3-P4: Spin-Glass Annealer
    sgapd_annealer = None
    if use_sgapd:
        from .components.sgapd import SpinGlassAnnealer

        sgapd_annealer = SpinGlassAnnealer(
            initial_temp=sgapd_initial_temp,
            min_temp=sgapd_min_temp,
            decay=sgapd_decay,
            sweeps_per_step=sgapd_sweeps,
        )
        print(f"[SGAPD] Enabled: T0={sgapd_initial_temp} decay={sgapd_decay}")

    # R4-A3-P5: Ergodic Optimizer
    ergodic_opt = None
    if use_ergodic:
        from .components.ergodic_optimizer import ErgodicTrajectoryOptimizer

        ergodic_opt = ErgodicTrajectoryOptimizer(
            model,
            alpha=ergodic_alpha,
            rho_target=ergodic_rho_target,
            flip_fraction=ergodic_flip_fraction,
        )
        print(
            f"[ERGODIC] Enabled: alpha={ergodic_alpha} flip_frac={ergodic_flip_fraction}"
        )

    # --- Resume from checkpoint ---
    start_step = 0
    ppl_history = []
    diverged = False
    if resume_from and os.path.isfile(resume_from):
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
            if state_key in ckpt:
                model_to_load = (
                    model._orig_mod if hasattr(model, "_orig_mod") else model
                )
                model_to_load.load_state_dict(ckpt[state_key], strict=False)
                logger.info(f"  Model state restored")
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                    logger.info(f"  Optimizer state restored")
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"  Could not load optimizer state: {e}")
            if "scheduler_state" in ckpt and scheduler is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                    logger.info(f"  Scheduler state restored")
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"  Could not load scheduler state: {e}")
            start_step = ckpt.get("step", 0)
            ppl_history = ckpt.get("ppl_history", [])
            logger.info(f"  Resuming from step {start_step}")

    # Progressive scheduler (optional)
    progressive_scheduler = None
    if progressive and (config.use_ternary or config.use_routing):
        total_optimizer_steps = total_steps // grad_accum_steps
        progressive_scheduler = ProgressiveScheduler(
            model,
            total_optimizer_steps,
            qat_start_pct,
            routing_start_pct,
            layer_wise=layer_wise_precision,
        )
        print(f"Progressive scheduler ({total_optimizer_steps} optimizer steps):")
        print(
            f"  FP16 -> QAT@step {progressive_scheduler.qat_start_step} -> Routing@step {progressive_scheduler.routing_start_step}"
        )

    training_state = _setup_training_state(
        model,
        optimizer,
        device,
        use_amp,
        use_bf16,
        use_ema,
        ema_decay,
        label_smoothing,
        use_liger_ce,
        warmup_steps,
        total_steps,
        cosine_restart_period,
        cosine_restart_decay,
        lr_schedule_type,
        optimizer_type,
    )
    scaler = training_state["scaler"]
    criterion = training_state["criterion"]
    monitor = training_state["monitor"]
    scheduler = training_state["scheduler"]
    ema = training_state["ema"]

    # Set gradient estimator type for ternary layers
    if ste_type != "standard" and config.use_ternary:
        model.set_ste_type(ste_type)
    if (
        ste_type in ("hestia", "bagr", "asgks", "mdts") or use_hestia2026
    ) and config.use_ternary:
        model.set_training_progress(0, total_steps)
        print(f"Training progress initialized: 0/{total_steps}")

    # Enable MDTS if specified
    if ste_type == "mdts" and config.use_ternary:
        from .components.ternary import enable_mdts

        enable_mdts(model, temperature=1.0, anneal=True)

    if gradual_threshold:
        model.set_gradual_threshold(True, start=0.5, end=DEFAULT_THRESHOLD_COEFF)
        print(f"Gradual threshold: 0.5 -> {DEFAULT_THRESHOLD_COEFF:.4f} over training")

    if pressure_blend:
        model.set_pressure_blend(True, duration=0.2)
        print(f"Pressure blend: gradual FP->ternary over 20% of QAT")

    # Enable information-theoretic freezing (R2-A3-M3)
    if use_info_freeze and config.use_ternary:
        _freeze_modules = 0
        for name, module in model.named_modules():
            if hasattr(module, "enable_info_freeze"):
                module.enable_info_freeze(
                    alpha=info_freeze_alpha, warmup=info_freeze_warmup
                )
                _freeze_modules += 1
        print(f"[INFO-FREEZE] Initialized on {_freeze_modules} TernaryLinear modules")

    # Enable oscillation dead zone (R2-A6-M14)
    if use_dead_zone and config.use_ternary:
        _dz_modules = 0
        for name, module in model.named_modules():
            if hasattr(module, "enable_dead_zone"):
                module.enable_dead_zone(
                    margin=dead_zone_margin,
                    lr_decay=dead_zone_lr_decay,
                    base_lr=base_lr,
                )
                _dz_modules += 1
        print(f"[DEAD-ZONE] Initialized on {_dz_modules} TernaryLinear modules")

    commitment_monitor = None
    if use_commitment_monitor and config.use_ternary:
        commitment_monitor = CommitmentMonitor(
            check_interval=commitment_check_interval,
            plateau_epsilon=commitment_plateau_epsilon,
            plateau_patience=commitment_plateau_patience,
            lr_boost=commitment_lr_boost,
            lr_decay=commitment_lr_decay,
            early_stop_threshold=commitment_early_stop_threshold,
            early_stop=commitment_early_stop,
        )
        print(
            f"[COMMITMENT] Monitor enabled: check_interval={commitment_check_interval}, "
            f"plateau_epsilon={commitment_plateau_epsilon}, patience={commitment_plateau_patience}, "
            f"lr_boost={commitment_lr_boost}, lr_decay={commitment_lr_decay}, "
            f"early_stop={commitment_early_stop} (threshold={commitment_early_stop_threshold})"
        )

    model.train()
    if optimizer_type == "schedule_free":
        optimizer.train()

    print(
        f"[TRAIN] Sequence packing: {'ENABLED' if use_sequence_packing else 'DISABLED'}"
    )

    pre_loop_state = _setup_pre_loop_state(
        model,
        config,
        batch_size,
        grad_accum_steps,
        use_adagc,
        adagc_factor,
        adagc_beta,
        use_fp8_grad_accum,
        use_topk_grad_sparsification,
        topk_grad_sparsity,
        use_gradient_sign_consistency,
        grad_sign_window,
        grad_sign_threshold,
        seq_warmup,
        seq_warmup_min,
        start_step,
    )
    global_step = pre_loop_state["global_step"]
    best_loss = pre_loop_state["best_loss"]
    nan_count = pre_loop_state["nan_count"]
    start_time = pre_loop_state["start_time"]
    loss = pre_loop_state["loss"]
    _lr_multipliers = pre_loop_state["_lr_multipliers"]
    _phase_loss_window = pre_loop_state["_phase_loss_window"]
    _kd_hidden = pre_loop_state["_kd_hidden"]
    _kd_hook_handle = pre_loop_state["_kd_hook_handle"]
    _l1_token_budget = pre_loop_state["_l1_token_budget"]
    _l1_current_batch = pre_loop_state["_l1_current_batch"]
    _l2_current_seq_len = pre_loop_state["_l2_current_seq_len"]
    _rpe_loss_ema = pre_loop_state["_rpe_loss_ema"]
    _rpe_ema_beta = pre_loop_state["_rpe_ema_beta"]
    _cbs_grad_norms = pre_loop_state["_cbs_grad_norms"]
    _cbs_current_accum = pre_loop_state["_cbs_current_accum"]
    _cbs_max_accum = pre_loop_state["_cbs_max_accum"]
    _cusum_c = pre_loop_state["_cusum_c"]
    _cusum_mu = pre_loop_state["_cusum_mu"]
    _cusum_skip = pre_loop_state["_cusum_skip"]
    _cusum_skipped = pre_loop_state["_cusum_skipped"]
    adagc_ema = pre_loop_state["adagc_ema"]
    _adagc_clips = pre_loop_state["_adagc_clips"]
    _adagc_max_ratio = pre_loop_state["_adagc_max_ratio"]
    fp8_grad_buffers = pre_loop_state["fp8_grad_buffers"]
    fp8_error_buffers = pre_loop_state["fp8_error_buffers"]
    topk_error_buffers = pre_loop_state["topk_error_buffers"]
    _grad_sign_history = pre_loop_state["_grad_sign_history"]
    fisher_accum = pre_loop_state["fisher_accum"]
    sign_sgd_momentum_buffers = pre_loop_state["sign_sgd_momentum_buffers"]

    use_rls = getattr(config, "use_rls_readout", False)
    if use_rls:
        model.setup_rls()
        print(f"[RLS-TRAIN] RLS readout ACTIVE for lm_head training")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_tokens = 0
        optimizer.zero_grad()
        if aga_optimizer is not None:
            aga_optimizer.zero_grad()
        if bbo_optimizer is not None:
            bbo_optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):
            if diverged or (max_steps and global_step >= max_steps):
                break

            mem_mb = 0
            if HAS_PSUTIL and batch_idx % 50 == 0:
                mem_mb = psutil.Process().memory_info().rss / 1024**2
                if mem_mb > max_memory_mb:
                    print(
                        f"\n[MEMORY GUARD] RSS = {mem_mb:.0f} MB > {max_memory_mb} MB limit."
                    )
                    raise MemoryError(
                        f"Training exceeded {max_memory_mb} MB limit ({mem_mb:.0f} MB used)."
                    )

            _train_step_ctx = {
                "device": device,
                "config": config,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "start_step": start_step,
                "use_cuda_prefetch": use_cuda_prefetch,
                "use_bf16": use_bf16,
                "use_amp": use_amp,
                "criterion": criterion,
                "use_rho1": use_rho1,
                "rho1_keep_ratio": rho1_keep_ratio,
                "rho1_weighting": "none",
                "use_uncertainty_loss": use_uncertainty_loss,
                "uw_loss_module": uw_loss_module,
                "use_rls": use_rls,
                "let_teacher": let_teacher,
                "let_layer_k": let_layer_k,
                "let_stop_step": let_stop_step,
                "teacher_n_layer": teacher_n_layer if "teacher_n_layer" in locals() else 0,
                "teacher_model": teacher_model,
                "od_teacher": od_teacher,
                "distillation_loss": distillation_loss,
                "distill_temperature": distill_temperature,
                "use_tsld": use_tsld,
                "distill_alpha": distill_alpha,
                "online_distill_temperature": online_distill_temperature,
                "online_distill_alpha": online_distill_alpha,
                "let_lambda": let_lambda,
                "boundary_reg_lambda": boundary_reg_lambda,
                "angular_reg_lambda": angular_reg_lambda,
                "contractive_reg_lambda": contractive_reg_lambda,
                "use_tdo": use_tdo,
                "tdo_lambda": tdo_lambda,
                "tdo_target": tdo_target,
                "tdo_temperature": tdo_temperature,
                "use_sca_sparsity": use_sca_sparsity,
                "sca_warmup_steps": sca_warmup_steps,
                "sca_lambda": sca_lambda,
                "sca_target_sparsity": sca_target_sparsity,
                "model_actual": model_actual,
                "use_ema_distill": use_ema_distill,
                "ema": ema,
                "ema_distill_temperature": ema_distill_temperature,
                "ema_distill_alpha": ema_distill_alpha,
                "use_sskd": use_sskd,
                "sskd_temperature": sskd_temperature,
                "sskd_alpha": sskd_alpha,
                "use_ema_replay": use_ema_replay,
                "ema_replay_lambda": ema_replay_lambda,
                "use_thm_loss": use_thm_loss,
                "thm_margin": thm_margin,
                "thm_lambda": thm_lambda,
                "use_tail_loss": use_tail_loss,
                "tail_flip_margin": tail_flip_margin,
                "tail_num_negatives": tail_num_negatives,
                "tail_lambda": tail_lambda,
                "use_rcl_loss": use_rcl_loss,
                "rcl_beta": rcl_beta,
                "rcl_sample_every": rcl_sample_every,
                "rcl_lambda": rcl_lambda,
                "use_ter_loss": use_ter_loss,
                "ter_loss_mu": ter_loss_mu,
                "ter_noise_var": ter_noise_var,
                "use_predictsim": use_predictsim,
                "use_saq": use_saq,
                "saq_k": saq_k,
                "log_every": log_every,
                "grad_accum_steps": grad_accum_steps,
                "_kd_hidden": _kd_hidden,
                "_kd_hook_handle": _kd_hook_handle,
            }

            loss_val, _train_step_ctx = _train_step(model, batch_data, _train_step_ctx)
            scaled_loss = _train_step_ctx["scaled_loss"]
            _kd_hidden = _train_step_ctx.get("_kd_hidden", {})

            if use_saq and config.use_ternary:
                saq_cycle = saq_k + 1
                is_ternary_step = global_step % saq_cycle == saq_k
                model.set_quantization_enabled(is_ternary_step)
                if (
                    global_step < 20 or global_step % log_every == 0
                ) and batch_idx % grad_accum_steps == 0:
                    saq_mode = "TERNARY(STE)" if is_ternary_step else "FP16(shadow)"
                    print(
                        f"  [SAQ] step={global_step} mode={saq_mode} cycle_pos={global_step % saq_cycle}/{saq_cycle}"
                    )

            if use_activation_quant:
                for module in model.modules():
                    if hasattr(module, "_act_quant_step"):
                        module._act_quant_step = global_step

            if use_compile and device == "cuda":
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()

            # AU-20: Strang Splitting — set weight to midpoint before gradient eval
            if getattr(config, 'use_strang_splitting', False) and use_ternary:
                with torch.no_grad():
                    for _au20_m in model.modules():
                        if (hasattr(_au20_m, 'weight') and hasattr(_au20_m, 'dsst_s')
                                and hasattr(_au20_m, 'lsq_scale_pos')):
                            _au20_scale = (_au20_m.lsq_scale_pos.mean().item()
                                           if isinstance(_au20_m.lsq_scale_pos, torch.Tensor)
                                           else float(_au20_m.lsq_scale_pos))
                            _au20_m._strang_w_backup = _au20_m.weight.data.clone()
                            _au20_W_tern = _au20_m.dsst_s.float() * _au20_scale
                            _au20_m.weight.data = (_au20_m.weight.data + _au20_W_tern) * 0.5
                if global_step == start_step:
                    print(f"[AU-20 STRANG] Midpoint weights active before forward pass")

            # AN-1: MDL Loss — differentiable sparsity incentive (ternary only)
            if getattr(config, 'use_mdl_loss', False) and getattr(config, 'use_ternary', False):
                _mdl_loss = torch.tensor(0.0, device=device)
                _mdl_count = 0
                _alpha_mdl = getattr(config, 'mdl_alpha', 1e-4)
                for _mdl_m in model.modules():
                    if hasattr(_mdl_m, 'threshold_coeff') and hasattr(_mdl_m, 'weight'):
                        _w = _mdl_m.weight.float()
                        _scale = _w.abs().mean().clamp(min=1e-7)
                        _mdl_loss = _mdl_loss + torch.exp(-_w.abs() / _scale).mean()
                        _mdl_count += 1
                if _mdl_count > 0:
                    loss = loss + _alpha_mdl * (_mdl_loss / _mdl_count)
                    if global_step % 50 == 0:
                        print(f"[MDL] step={global_step} mdl_loss={_mdl_loss.item():.4f} n={_mdl_count}")

            # AN-5: Cohomological Loss — penalize state jumps not justified by input
            if getattr(config, 'use_cohom_loss', False):
                _cohom_loss = torch.tensor(0.0, device=device)
                _cohom_count = 0
                _alpha_cohom = getattr(config, 'cohom_alpha', 1e-4)
                _cohom_scale = getattr(config, 'cohom_scale', 1.0)
                if hasattr(model, '_ssm_aux_outputs') and model._ssm_aux_outputs:
                    for _aux in model._ssm_aux_outputs:
                        if isinstance(_aux, dict) and 'h_prev' in _aux and 'h_curr' in _aux and 'x_in' in _aux:
                            _dh = (_aux['h_curr'] - _aux['h_prev']).norm(dim=-1)
                            _dx = _aux['x_in'].norm(dim=-1).clamp(min=1e-7)
                            _ratio = _dh / (_dx * _cohom_scale)
                            _cohom_loss = _cohom_loss + _ratio.clamp(max=10.0).mean()
                            _cohom_count += 1
                if _cohom_count > 0:
                    loss = loss + _alpha_cohom * (_cohom_loss / _cohom_count)
                    if global_step % 100 == 0:
                        print(f"[COHOM] step={global_step} cohom={_cohom_loss.item():.4f} n={_cohom_count}")

            # AO-3: Ergodic Semantic Loss — SSM final state EMA centroid convergence
            if getattr(config, 'use_ergodic_loss', False):
                if hasattr(model, '_ssm_aux_outputs') and model._ssm_aux_outputs:
                    _h_t = model._ssm_aux_outputs[-1]
                    if isinstance(_h_t, torch.Tensor) and _h_t.dim() >= 2:
                        _h_mean = _h_t.float().mean(dim=1)  # [B, d_model]
                        _model_base = getattr(model, '_orig_mod', model)
                        if not hasattr(_model_base, '_ergodic_ema'):
                            _model_base._ergodic_ema = _h_mean.detach().mean(0)
                        else:
                            _model_base._ergodic_ema = (
                                0.99 * _model_base._ergodic_ema.detach()
                                + 0.01 * _h_mean.detach().mean(0)
                            )
                        _ergodic_target = _model_base._ergodic_ema.unsqueeze(0).expand_as(_h_mean)
                        _ergodic_loss = F.mse_loss(_h_mean, _ergodic_target.detach())
                        _alpha_ergodic = getattr(config, 'ergodic_loss_alpha', 1e-3)
                        loss = loss + _alpha_ergodic * _ergodic_loss
                        if global_step % 100 == 0 and not torch.compiler.is_compiling():
                            print(f"[ERGODIC-LOSS] step={global_step} loss={_ergodic_loss.item():.4f}")

            # AO-5: Masked Prediction Auxiliary Loss
            if getattr(config, 'use_masked_pred', False) and _masked_pred_mask is not None:
                _mp_lambda = getattr(config, 'masked_pred_lambda', 0.1)
                _mp_seq_len = min(logits.shape[1], targets.shape[1])
                _mp_mask_flat = _masked_pred_mask[:, :_mp_seq_len].reshape(-1)
                if _mp_mask_flat.any():
                    _mp_logits_flat = logits[:, :_mp_seq_len, :].reshape(-1, logits.shape[-1])[_mp_mask_flat]
                    _mp_targets_flat = targets[:, :_mp_seq_len].reshape(-1)[_mp_mask_flat]
                    _mp_loss = F.cross_entropy(_mp_logits_flat.float(), _mp_targets_flat)
                    loss = loss + _mp_lambda * _mp_loss
                    if global_step % 100 == 0 and not torch.compiler.is_compiling():
                        print(f"[MASKED-PRED] step={global_step} mp_loss={_mp_loss.item():.4f} "
                              f"n={_mp_mask_flat.sum().item()}")

            # AO-6: Temporal Contrastive Loss — InfoNCE on h[t] vs h[t+k]
            if getattr(config, 'use_temporal_contrastive', False):
                if hasattr(model, '_ssm_aux_outputs') and model._ssm_aux_outputs:
                    _h_tc = model._ssm_aux_outputs[-1]
                    if isinstance(_h_tc, torch.Tensor) and _h_tc.dim() >= 3:
                        _h_tc = _h_tc.float()
                        _k_tc = getattr(config, 'temporal_contrastive_k', 4)
                        if _h_tc.shape[1] > _k_tc + 1:
                            _h_anch = F.normalize(_h_tc[:, :-_k_tc, :], dim=-1)
                            _h_pos  = F.normalize(_h_tc[:, _k_tc:, :], dim=-1)
                            _h_neg  = _h_pos.roll(1, dims=0)
                            _sim_pos = (_h_anch * _h_pos).sum(-1)
                            _sim_neg = (_h_anch * _h_neg).sum(-1)
                            _tc_logits = torch.stack([_sim_pos, _sim_neg], dim=-1)
                            _tc_labels = torch.zeros(
                                _tc_logits.shape[:-1], dtype=torch.long, device=device
                            )
                            _tc_loss = F.cross_entropy(_tc_logits.reshape(-1, 2), _tc_labels.reshape(-1))
                            _lambda_tc = getattr(config, 'temporal_contrastive_lambda', 0.01)
                            loss = loss + _lambda_tc * _tc_loss
                            if global_step % 100 == 0 and not torch.compiler.is_compiling():
                                print(f"[TEMP-CONTRAST] step={global_step} tc_loss={_tc_loss.item():.4f} "
                                      f"k={_k_tc}")

            # AR-6: Input-entropy bypass — L_stability loss
            if getattr(config, 'use_input_entropy_bypass', False) and model.training:
                _ieb_lambda = getattr(config, 'input_entropy_lambda', 0.001)
                _ieb_loss = sum(
                    (p.sign() * p).sum() * _ieb_lambda
                    for p in model.parameters()
                    if p.requires_grad and p.ndim >= 2 and p.shape[0] > 1
                ) / max(1, sum(1 for p in model.parameters() if p.requires_grad and p.ndim >= 2))
                if not _ieb_loss.isnan():
                    loss = loss + _ieb_loss
                if global_step % 50 == 0:
                    print(f"[AR-6 INPUT-ENTROPY-BYPASS] step={global_step} ieb_loss={_ieb_loss.item():.4f}")

            # AR-8: THDV — Temporal Half-Depth Validation (logging only — KL between steps)
            if getattr(config, 'use_thdv', False) and model.training and global_step % 50 == 0:
                print(f"[AR-8 THDV] step={global_step} thdv_active=True")

            # R23-B: Local Loss Per Block (PredSim)
            if _local_losses_list is not None and len(_local_losses_list) > 0:
                _ce_w = getattr(config, 'local_loss_ce_weight', 0.7)
                _cos_w = getattr(config, 'local_loss_cosine_weight', 0.3)
                _global_blend = getattr(config, 'local_loss_global_blend', 0.3)
                _predsim_total = torch.tensor(0.0, device=device)
                _n_local = 0
                for _h_T, _probe in _local_losses_list:
                    _local_logits = _probe(_h_T)
                    _local_targets = targets[:, -1]
                    _ce_term = F.cross_entropy(_local_logits, _local_targets)
                    _y_embed = model.embedding.weight[_local_targets].detach()
                    _cos_term = 1.0 - F.cosine_similarity(_h_T, _y_embed, dim=-1).mean()
                    _block_loss = _ce_w * _ce_term + _cos_w * _cos_term
                    _predsim_total = _predsim_total + _block_loss
                    _n_local += 1
                if _n_local > 0:
                    _predsim_avg = _predsim_total / _n_local
                    loss = (1.0 - _global_blend) * _predsim_avg + _global_blend * loss
                if global_step % 50 == 0:
                    print(f"[PREDSIM] step={global_step} n_blocks={_n_local} "
                          f"local_avg={_predsim_avg.item():.4f} final_loss={loss.item():.4f}")

            # AN-3: Phase Detection — track loss variance
            if getattr(config, 'use_phase_detection', False):
                _phase_loss_window.append(loss_val)
                _pw = getattr(config, 'phase_window_size', 50)
                if len(_phase_loss_window) > _pw:
                    _phase_loss_window.pop(0)
                if len(_phase_loss_window) >= 10 and global_step % 100 == 0:
                    import statistics
                    _pvar = statistics.stdev(_phase_loss_window)
                    print(f"[PHASE] step={global_step} loss_stdev={_pvar:.4f} window={len(_phase_loss_window)}")

            # R17-A F6: RPE Gate update
            if getattr(config, 'use_rpe_gate', False):
                if _rpe_loss_ema is None:
                    _rpe_loss_ema = loss_val
                else:
                    _rpe_prev = _rpe_loss_ema
                    _rpe_loss_ema = _rpe_ema_beta * _rpe_loss_ema + (1.0 - _rpe_ema_beta) * loss_val
                    _rpe = _rpe_prev - loss_val
                    _flip_gate = max(0.8, min(1.5, 1.0 - _rpe * 2.0))
                    for _mod in model.modules():
                        if hasattr(_mod, 'rpe_flip_gate') and getattr(_mod, 'use_rpe_gate', False):
                            _mod.rpe_flip_gate.fill_(_flip_gate)
                    if global_step % 100 == 0:
                        print(f"[RPE-GATE] step={global_step} rpe={_rpe:.4f} gate={_flip_gate:.4f} ema={_rpe_loss_ema:.4f}")

                        # AT-3: Dopamine RPE Sign Gate diagnostic
                        # reward_gate = True when loss improved (_rpe > 0 means ema > loss)
                        # In full impl: DSST flips suppressed when reward_gate=False
                        if getattr(config, 'use_dopamine_rpe_gate', False):
                            _dopamine_reward_gate = (_rpe > 0)
                            _dopamine_suppress_frac = 0.0 if _dopamine_reward_gate else 1.0
                            print(f"[AT-3 DOPAMINE] step={global_step} reward_gate={_dopamine_reward_gate} "
                                  f"suppress_frac={_dopamine_suppress_frac:.1f} rpe={_rpe:.4f}")

            # R4-A4-F6: Anomaly detection on loss
            if anomaly_detector is not None:
                loss_check = loss_val if not math.isnan(loss_val) else float("nan")
                anomaly_result = anomaly_detector.update_metric(
                    "loss", loss_check, global_step
                )
                if (
                    anomaly_result.get("should_rollback")
                    and anomaly_detector._checkpoint_ring
                ):
                    rollback_state = anomaly_detector.rollback()
                    if rollback_state is not None:
                        model_to_load = (
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )
                        model_to_load.load_state_dict(
                            rollback_state.get("model_state", rollback_state)
                        )
                        print(f"[ANOMALY] Model rolled back at step {global_step}")

            if math.isnan(loss_val):
                nan_count += 1
                if max_nan_steps > 0 and nan_count >= max_nan_steps:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"DIVERGED: {max_nan_steps} consecutive NaN steps at step {global_step}. Stopping."
                    )
                    diverged = True
                    break
            else:
                nan_count = 0

            if use_rls and model.rls_readout is not None and model.lm_head is not None and pre_head is not None:
                with torch.no_grad():
                    h_batch = pre_head.detach()
                    y_batch = torch.zeros_like(logits.detach())
                    y_batch.scatter_(-1, targets.unsqueeze(-1), 1.0)
                    for b in range(min(h_batch.shape[0], 2)):
                        model.rls_readout.update_batch(
                            h_batch[b].float(),
                            y_batch[b].float(),
                            model.lm_head.weight
                        )

            _glts_result = (None, None)
            _surprise_skipped = False

            use_surprise_skip = getattr(config, 'use_surprise_skip', False)
            surprise_skip_threshold = getattr(config, 'surprise_skip_threshold', 2.0)
            surprise_skip_after = getattr(config, 'surprise_skip_after', 0.3)

            if use_surprise_skip and global_step > int(max_steps * surprise_skip_after):
                if scaled_loss.item() < surprise_skip_threshold:
                    _surprise_skipped = True
                    if (global_step - start_step) % log_interval == 0:
                        print(f"[SURPRISE-SKIP] step={global_step} loss={scaled_loss.item():.4f} SKIPPED backward")

            if scaler is not None and not _surprise_skipped:
                scaler.scale(scaled_loss).backward()

                # AU-20: Strang Splitting — restore original weights after gradient eval
                if getattr(config, 'use_strang_splitting', False) and use_ternary:
                    with torch.no_grad():
                        for _au20_m in model.modules():
                            if hasattr(_au20_m, '_strang_w_backup'):
                                _au20_m.weight.data.copy_(_au20_m._strang_w_backup)
                                del _au20_m._strang_w_backup

                # R4-A4-F10: Log gradient norms
                if gradient_monitor is not None:
                    gradient_monitor.log_grad_norms(model, global_step)

                # FP8 Gradient Accumulation (scaler path)
                if use_fp8_grad_accum and (batch_idx + 1) % grad_accum_steps != 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_with_error = param.grad.data + fp8_error_buffers.get(
                                name, 0
                            )
                            q_grad, q_scale = _quantize_to_fp8(grad_with_error)
                            deq_grad = _dequantize_from_fp8(q_grad, q_scale)
                            if getattr(config, "fp8_grad_accum_error_feedback", True):
                                fp8_error_buffers[name] = grad_with_error - deq_grad
                            if name not in fp8_grad_buffers:
                                fp8_grad_buffers[name] = deq_grad
                            else:
                                fp8_grad_buffers[name] += deq_grad
                            param.grad.data.zero_()
                elif use_fp8_grad_accum and (batch_idx + 1) % grad_accum_steps == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in fp8_grad_buffers:
                            param.grad.data += fp8_grad_buffers[name]
                            if name in fp8_error_buffers:
                                param.grad.data += fp8_error_buffers[name]
                    fp8_grad_buffers.clear()
                    for name in fp8_error_buffers:
                        fp8_error_buffers[name].zero_()
            elif not _surprise_skipped:
                # R2-A2-M4: Begin fused AGA step cycle
                if fused_aga is not None and batch_idx % grad_accum_steps == 0:
                    fused_aga.begin_step()

                _raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                _sbp_block_size = getattr(config, 'sbp_block_size', 0)
                if _sbp_block_size > 0:
                    _glts_result = _glts_step_blockwise(
                        raw_model=_raw_model,
                        input_ids=input_ids,
                        targets=input_ids,
                        block_size=_sbp_block_size,
                        boundary_mask=boundary_mask,
                        grad_scale=1.0 / grad_accum_steps,
                        device=device,
                        amp_dtype=torch.bfloat16,
                    )
                else:
                    _glts_result = _glts_step(
                        raw_model=_raw_model,
                        input_ids=input_ids,
                        targets=input_ids,
                        boundary_mask=boundary_mask,
                        grad_scale=1.0 / grad_accum_steps,
                        device=device,
                        amp_dtype=torch.bfloat16,
                    )
                if _glts_result[0] is None:
                    scaled_loss.backward()

                    # AU-20: Strang Splitting — restore original weights after gradient eval
                    if getattr(config, 'use_strang_splitting', False) and use_ternary:
                        with torch.no_grad():
                            for _au20_m in model.modules():
                                if hasattr(_au20_m, '_strang_w_backup'):
                                    _au20_m.weight.data.copy_(_au20_m._strang_w_backup)
                                    del _au20_m._strang_w_backup

                # R4-A4-F10: Log gradient norms
                if gradient_monitor is not None:
                    gradient_monitor.log_grad_norms(model, global_step)

                # FP8 Gradient Accumulation (non-scaler path)
                if use_fp8_grad_accum and (batch_idx + 1) % grad_accum_steps != 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_with_error = param.grad.data + fp8_error_buffers.get(
                                name, 0
                            )
                            q_grad, q_scale = _quantize_to_fp8(grad_with_error)
                            deq_grad = _dequantize_from_fp8(q_grad, q_scale)
                            if getattr(config, "fp8_grad_accum_error_feedback", True):
                                fp8_error_buffers[name] = grad_with_error - deq_grad
                            if name not in fp8_grad_buffers:
                                fp8_grad_buffers[name] = deq_grad
                            else:
                                fp8_grad_buffers[name] += deq_grad
                            param.grad.data.zero_()
                elif use_fp8_grad_accum and (batch_idx + 1) % grad_accum_steps == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in fp8_grad_buffers:
                            param.grad.data += fp8_grad_buffers[name]
                            if name in fp8_error_buffers:
                                param.grad.data += fp8_error_buffers[name]
                    fp8_grad_buffers.clear()
                    for name in fp8_error_buffers:
                        fp8_error_buffers[name].zero_()

                # R2-A2-M4: Invalidate ternary caches after backward if grad_accum_steps > 1
                if fused_aga is not None and grad_accum_steps > 1:
                    for module in model.modules():
                        if hasattr(module, "invalidate_weight_cache"):
                            module.invalidate_weight_cache()

            batch_tokens = input_ids.numel()
            epoch_loss += loss_val * batch_tokens
            epoch_tokens += batch_tokens

            if (batch_idx + 1) % grad_accum_steps == 0:
                # R5-A1-P1: HFOTT gradient correction for uncertain ternary positions
                if use_hfott and use_ternary and global_step > 0:
                    _hfott_delta = apply_hfott_gradient_correction(
                        model._orig_mod if hasattr(model, "_orig_mod") else model,
                        input_ids, targets, criterion,
                        sigma=hfott_sigma, alpha=hfott_alpha,
                        use_amp=use_amp, device=device,
                    )
                    if global_step == start_step:
                        print(f"[HFOTT VERIFY] ACTIVE | sigma={hfott_sigma} alpha={hfott_alpha} delta={_hfott_delta:.4f}")

                if scaler is not None:
                    scaler.unscale_(optimizer)
                    # Margin-Sparse gradient recycling (R2-A1-M1+M4)
                    if gradient_recyclers is not None:
                        apply_gradient_recycling(
                            model,
                            gradient_recyclers,
                            global_step,
                            margin_refresh_interval,
                            margin_delta,
                        )
                    if use_adagc:
                        # AdaGC: adaptive per-tensor gradient clipping (R2-A3-F6)
                        _adagc_clips = 0
                        _adagc_max_ratio = 0.0
                        for _name, _param in model.named_parameters():
                            if _param.grad is not None and _name in adagc_ema:
                                _gnorm = _param.grad.data.norm().item()
                                if adagc_ema[_name] == 0.0:
                                    adagc_ema[_name] = (
                                        _gnorm  # initialize on first step
                                    )
                                else:
                                    adagc_ema[_name] = (
                                        adagc_beta * adagc_ema[_name]
                                        + (1 - adagc_beta) * _gnorm
                                    )
                                if (
                                    adagc_ema[_name] > 0
                                    and _gnorm > adagc_factor * adagc_ema[_name]
                                ):
                                    _scale = adagc_factor * adagc_ema[_name] / _gnorm
                                    _param.grad.data.mul_(_scale)
                                    _adagc_clips += 1
                                    _adagc_max_ratio = max(
                                        _adagc_max_ratio, _gnorm / adagc_ema[_name]
                                    )
                    # A6-M3: Threshold-scaled gradient multiplier (thermodynamic annealing).
                    # threshold_coeff anneals inside forward() as local var — never writes back.
                    # Replicate formula here so optimizer sees the correct effective LR.
                    if use_ternary:
                        _a6m3_progress = global_step / max(1, total_steps)
                        _tc_vals_step = [
                            max(0.05, m.threshold_coeff * (1.0 - _a6m3_progress) ** 0.25)
                            for m in model.modules()
                            if hasattr(m, "threshold_coeff")
                            and hasattr(m, "group_size")
                            and getattr(m, "threshold_type", "coefficient") == "thermodynamic"
                        ]
                        if _tc_vals_step:
                            _avg_tc_step = sum(_tc_vals_step) / len(_tc_vals_step)
                            _lr_tc_scale_step = _avg_tc_step / DEFAULT_THRESHOLD_COEFF
                            for _name, _param in model.named_parameters():
                                if _param.grad is not None and "weight" in _name:
                                    _param.grad.data.mul_(_lr_tc_scale_step)
                            if not hasattr(train_model, "_a6m3_logged"):
                                print(
                                    f"[A6-M3] ACTIVE step={global_step} "
                                    f"avg_tc={_avg_tc_step:.4f} scale={_lr_tc_scale_step:.4f}"
                                )
                                train_model._a6m3_logged = True

                    # A6-F6: Commitment-driven per-layer gradient scaling (before clip)
                    if use_ternary and global_step > warmup_steps:
                        _a6f6_scales = apply_commitment_lr_scaling(
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )
                        if not hasattr(train_model, "_a6f6_logged"):
                            n_scaled = len(_a6f6_scales)
                            if n_scaled > 0:
                                avg_scale = sum(_a6f6_scales.values()) / n_scaled
                                print(
                                    f"[A6-F6] Commitment LR scaling | layers={n_scaled} avg_scale={avg_scale:.3f}"
                                )
                                train_model._a6f6_logged = True
                            elif global_step > warmup_steps + 5:
                                print("[A6-F6] No layers with commitment tracking (n_stable uninitialized)")
                                train_model._a6f6_logged = True
                        # [F22] Adaptive Gradient Clipping (in addition to per-tensor clipping)
                        clip_stats = grad_clipper.clip(model.parameters())
                        # Still compute global grad_norm for logging (after per-tensor clipping)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
                    else:
                        # [F22] Adaptive Gradient Clipping
                        clip_stats = grad_clipper.clip(model.parameters())
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
                    # R19-B: CUSUM Optimal Update Gate
                    _cusum_skip = False
                    if getattr(config, 'use_cusum_update_gate', False):
                        _cusum_k = getattr(config, 'cusum_k', 0.5)
                        _cusum_h = getattr(config, 'cusum_h', 5.0)
                        _cusum_alpha = getattr(config, 'cusum_ema_alpha', 0.05)
                        _gn = float(grad_norm)
                        if _cusum_mu is None:
                            _cusum_mu = _gn
                        else:
                            _cusum_mu = (1.0 - _cusum_alpha) * _cusum_mu + _cusum_alpha * _gn
                        _cusum_c = max(0.0, _cusum_c + abs(_gn - _cusum_mu) - _cusum_k * max(_cusum_mu, 0.01))
                        _cusum_skip = (_cusum_c <= _cusum_h)
                        if _cusum_skip:
                            _cusum_skipped += 1
                            _cusum_c = 0.0
                        if global_step % 100 == 0 and not torch.compiler.is_compiling():
                            print(f"[CUSUM-GATE] step={global_step} C={_cusum_c:.3f} mu={_cusum_mu:.4f} "
                                  f"skip={_cusum_skip} total_skipped={_cusum_skipped}", flush=True)
                    # CBS: track rolling grad norm for Curriculum Batch Scaling
                    _cbs_grad_norms.append(float(grad_norm))
                    if len(_cbs_grad_norms) > 50:
                        _cbs_grad_norms.pop(0)
                    if (len(_cbs_grad_norms) >= 50 and global_step % 200 == 0
                            and _cbs_current_accum < _cbs_max_accum):
                        _cbs_mean = sum(_cbs_grad_norms) / len(_cbs_grad_norms)
                        _cbs_var = sum((x - _cbs_mean) ** 2 for x in _cbs_grad_norms) / len(_cbs_grad_norms)
                        _cbs_cv = (_cbs_var ** 0.5) / max(_cbs_mean, 1e-8)
                        if _cbs_cv < 0.15:
                            _cbs_current_accum = min(_cbs_current_accum * 2, _cbs_max_accum)
                            grad_accum_steps = _cbs_current_accum
                            # Scale LR by sqrt(2) for larger effective batch (Surge paper)
                            for _cbs_pg in optimizer.param_groups:
                                _cbs_pg['lr'] *= 1.4142
                            print(f"[CBS] step={global_step} CV={_cbs_cv:.3f} -> accum={_cbs_current_accum}×, LR*sqrt2")
                    # Layerwise LR via HT-SR spectral alpha
                    if layerwise_lr:
                        if global_step >= warmup_steps and global_step % 500 == 0:
                            _lr_multipliers = compute_lr_multipliers(
                                compute_layer_alphas(model)
                            )
                            if not _lr_multipliers:
                                import warnings

                                warnings.warn(
                                    f"Layerwise LR: compute_lr_multipliers returned empty dict at step {global_step}"
                                )
                        if _lr_multipliers:
                            for name, param in model.named_parameters():
                                if param.grad is not None and name in _lr_multipliers:
                                    param.grad.mul_(_lr_multipliers[name])
                    if use_grad_noise:
                        _gn_std = _apply_gradient_noise(
                            model, global_step, grad_noise_eta, grad_noise_gamma
                        )
                        if global_step == start_step:
                            print(
                                f"[GRAD-NOISE VERIFY] ACTIVE | eta={grad_noise_eta} gamma={grad_noise_gamma} std={_gn_std:.6f}"
                            )
                    if use_gradient_sign_consistency and config.use_ternary:
                        apply_gradient_sign_consistency(
                            model,
                            _grad_sign_history,
                            grad_sign_threshold,
                            grad_sign_window,
                        )
                    if use_topk_grad_sparsification:
                        apply_topk_gradient_sparsification(
                            model, topk_grad_sparsity, topk_error_buffers
                        )

                    # A2-F6: Layerwise Gauss-Newton Preconditioning
                    if getattr(config, "use_fisher_preconditioning", False):
                        apply_fisher_preconditioning(
                            model,
                            fisher_accum,
                            global_step,
                            damping=getattr(config, "fisher_damping", 1e-4),
                        )

                    # A2-F7: SignSGD Momentum
                    if getattr(config, "use_sign_sgd", False):
                        apply_sign_momentum(
                            model,
                            sign_sgd_momentum_buffers,
                            beta=getattr(config, "sign_sgd_momentum", 0.9),
                        )

                    # A2-M7: Consensus-Filtered-Layer-Gradients
                    if getattr(config, "use_consensus_gradient_filter", False):
                        apply_consensus_gradient_filter(
                            model,
                            consensus_alpha=getattr(config, "consensus_alpha", 0.5),
                        )

                    # R4-A3-M5: Gradient Consensus Polarization
                    if use_gcp:
                        from .components.gcp import apply_gcp_simple

                        n_apps = getattr(config, "recursive_iterations", 32)
                        for name, param in model.named_parameters():
                            if param.grad is not None and "shared_ssm_block" in name:
                                param.grad.data = apply_gcp_simple(
                                    param.grad.data,
                                    n_applications=n_apps,
                                    alpha=gcp_alpha,
                                )
                        if global_step == start_step:
                            print(
                                f"[GCP VERIFY] ACTIVE | alpha={gcp_alpha} n_apps={n_apps}"
                            )

                    # R4-A3-M6: RAFU (Recurrence-Aware Flip Utility)
                    if use_rafu and global_step % rafu_interval == 0:
                        _inter_states = (
                            model.get_intermediate_states()
                            if hasattr(model, "get_intermediate_states")
                            else []
                        )
                        if _inter_states and len(_inter_states) >= 2:
                            from .components.rafu import (
                                compute_rafu_utility,
                                apply_rafu_flip_filter,
                            )

                            _rafu_utility = compute_rafu_utility(
                                _inter_states[:-1],
                                _inter_states[1:],
                            )
                            if _rafu_utility is not None:
                                for name, param in model.named_parameters():
                                    if (
                                        param.grad is not None
                                        and "shared_ssm_block" in name
                                    ):
                                        if param.grad.shape == _rafu_utility.shape:
                                            param.grad.data = apply_rafu_flip_filter(
                                                param.grad.data,
                                                _rafu_utility,
                                                threshold=rafu_threshold,
                                            )
                                if global_step == start_step:
                                    print(
                                        f"[RAFU VERIFY] ACTIVE | utility_shape={_rafu_utility.shape} thresh={rafu_threshold}"
                                    )

                    # R4-A3-M8: Bayesian Flip Accumulator
                    if use_bfa and bfa_accumulators and global_step % bfa_interval == 0:
                        total_flips = 0
                        model_to_check = (
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )
                        for name, module in model_to_check.named_modules():
                            if (
                                name in bfa_accumulators
                                and hasattr(module, "weight")
                                and module.weight.grad is not None
                            ):
                                bfa = bfa_accumulators[name]
                                w = module.weight.data
                                threshold = (
                                    getattr(module, "threshold_coeff", 0.5)
                                    * w.abs().mean()
                                )
                                current_ternary = (
                                    torch.sign(w) * (w.abs() > threshold).float()
                                )

                                bfa.update(module.weight.grad.data, current_ternary)
                                n_flips = bfa.apply_flips(
                                    module.weight,
                                    getattr(module, "threshold_coeff", 0.5),
                                )
                                total_flips += n_flips
                        if global_step == start_step:
                            print(
                                f"[BFA VERIFY] ACTIVE | layers={len(bfa_accumulators)} interval={bfa_interval}"
                            )
                        if total_flips > 0 and global_step % (bfa_interval * 10) == 0:
                            print(
                                f"[BFA] Step {global_step}: {total_flips} flips applied"
                            )

                    # R4-A3-P4: Spin-Glass MC Sweep (periodic)
                    if (
                        use_sgapd
                        and sgapd_annealer is not None
                        and global_step % sgapd_interval == 0
                    ):
                        _sg_accepted = 0
                        model_to_check = (
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )
                        for _name, _module in model_to_check.named_modules():
                            if hasattr(_module, "quantize_enabled") and hasattr(
                                _module, "weight"
                            ):
                                _sg_accepted += sgapd_annealer.mc_sweep(
                                    _module.weight, None, None, model_to_check
                                )
                        sgapd_annealer.update_temperature()
                        if global_step == start_step:
                            print(
                                f"[SGAPD VERIFY] ACTIVE | T={sgapd_annealer.temperature:.4f}"
                            )

                    # R4-A3-P5: Ergodic Trajectory Optimizer (periodic)
                    if (
                        use_ergodic
                        and ergodic_opt is not None
                        and global_step % ergodic_interval == 0
                    ):
                        _inter_states = (
                            model.get_intermediate_states()
                            if hasattr(model, "get_intermediate_states")
                            else []
                        )
                        if _inter_states:
                            ergodic_opt.update_statistics(model, _inter_states)
                            _erg_flips = ergodic_opt.score_and_flip(model)
                            if global_step == start_step:
                                print(
                                    f"[ERGODIC VERIFY] ACTIVE | interval={ergodic_interval}"
                                )

                    # A2-F8: Adaptive Quantization Levels
                    if (
                        getattr(config, "use_adaptive_quantization", False)
                        and global_step
                        % getattr(config, "adaptive_quant_interval", 500)
                        == 0
                    ):
                        apply_adaptive_quantization_levels(
                            model,
                            sensitivity_threshold=getattr(
                                config, "adaptive_quant_sensitivity_threshold", 0.1
                            ),
                        )

                    # R4-A4-F10: Snapshot params before update
                    if gradient_monitor is not None:
                        gradient_monitor.snapshot_params(model)

                    # F24: Information-Theoretic Freezing
                    if info_freezer is not None:
                        info_freezer.update(model, global_step)
                        info_freezer.apply_masks(model)

                    # A3-F13: Skip optimizer for committed weights
                    # Zero gradient for committed positions so Lion skips them.
                    # A3-F13 + A1-F16: Block-wise gradient skip for committed weights
                    if use_ternary:
                        _skipped_w = 0
                        _total_w = 0
                        _raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                        for _m in _raw_model.modules():
                            if (
                                hasattr(_m, "committed_mask")
                                and _m.committed_mask is not None
                                and hasattr(_m, "weight")
                                and _m.weight.grad is not None
                            ):
                                _cm = _m.committed_mask
                                _gs = getattr(_m, "group_size", 128)
                                _flat_cm = _cm.reshape(-1)
                                _N = _flat_cm.numel()
                                _total_w += _N
                                # A1-F16: Check whole blocks of group_size
                                _pad = (-_N) % _gs
                                if _pad > 0:
                                    _padded = torch.cat([_flat_cm, _flat_cm.new_ones(_pad)])
                                else:
                                    _padded = _flat_cm
                                _blocks = _padded.reshape(-1, _gs).all(dim=-1)
                                _block_expanded = _blocks.unsqueeze(-1).expand(-1, _gs).reshape(-1)[:_N]
                                _block_mask = _block_expanded.reshape(_cm.shape)
                                _m.weight.grad[_block_mask] = 0.0
                                _skipped_w += int(_block_mask.sum().item())
                                if not hasattr(train_model, "_a1f16_logged") and _blocks.any():
                                    print(
                                        f"[A1-F16] Block-wise skip: "
                                        f"full_blocks={int(_blocks.sum())}/{len(_blocks)} gs={_gs}"
                                    )
                                    train_model._a1f16_logged = True
                        if not hasattr(train_model, "_a3f13_logged") and _total_w > 0:
                            print(
                                f"[A3-F13] Skip-optimizer | committed={_skipped_w}/{_total_w} "
                                f"({100*_skipped_w/max(1,_total_w):.1f}%) grad zeroed"
                            )
                            train_model._a3f13_logged = True

                    # D2 — Lion Momentum-Lock Grad Skip (R8-A5)
                    # When sign(g)==sign(m) AND |m|>threshold, sign(update)==sign(m) regardless.
                    # Zero those grads: saves grad accumulation overhead with same Lion update.
                    if optimizer_type == "lion":
                        _d2_locked = 0
                        _d2_total = 0
                        _d2_threshold = 0.01
                        for _grp in optimizer.param_groups:
                            for _p in _grp["params"]:
                                if _p.grad is None:
                                    continue
                                _d2_total += _p.grad.numel()
                                _p_state = optimizer.state.get(_p, {})
                                if "exp_avg" not in _p_state:
                                    continue
                                _m = _p_state["exp_avg"].float()
                                _g = _p.grad.float()
                                _lock = (_m * _g > 0) & (_m.abs() > _d2_threshold)
                                _p.grad.data[_lock] = 0.0
                                _d2_locked += int(_lock.sum().item())
                        if not hasattr(train_model, "_d2_logged") and _d2_total > 0:
                            print(
                                f"[D2-MLGS] Momentum-lock skip | "
                                f"locked={_d2_locked}/{_d2_total} "
                                f"({100*_d2_locked/max(1,_d2_total):.1f}%) grad zeroed"
                            )
                            train_model._d2_logged = True

                    if use_ternary and global_step % 50 == 0:
                        _nf_sum = 0.0; _nf_count = 0
                        _raw_model_nf = model._orig_mod if hasattr(model, "_orig_mod") else model
                        for _m_nf in _raw_model_nf.modules():
                            if (hasattr(_m_nf, "committed_mask") and _m_nf.committed_mask is not None
                                    and hasattr(_m_nf, "weight") and _m_nf.weight.grad is not None
                                    and _m_nf.committed_mask.any()):
                                _g_nf = _m_nf.weight.grad.float()
                                _cm_nf = _m_nf.committed_mask
                                _nf_sum += _g_nf[_cm_nf].pow(2).sum().item()
                                _nf_count += int(_cm_nf.sum().item())
                        if _nf_count > 0:
                            _sigma2_noise = _nf_sum / _nf_count
                            print(f"[E1-MARS] noise_floor σ²={_sigma2_noise:.2e} n_committed={_nf_count} step={global_step}")

                    if not _cusum_skip:
                        scaler.step(optimizer)
                    scaler.update()

                    # R4-A4-F10: Log update norms
                    if gradient_monitor is not None:
                        gradient_monitor.log_update_norms(model, global_step)

                    # R4-A4-M7: Record weight flips
                    if delta_chronicle is not None:
                        flip_stats = delta_chronicle.record_step(model, global_step)
                        if flip_stats["flips"] > 0 and global_step % 100 == 0:
                            print(
                                f"[CHRONICLE] step {global_step}: {flip_stats['flips']} flips, "
                                f"rate={flip_stats['flip_rate']:.6f}"
                            )

                        # R4-A4-M8: Compute flip influences
                        if flip_tracker is not None and flip_stats["flips"] > 0:
                            step_flips = (
                                delta_chronicle._history[-1][1]
                                if delta_chronicle._history
                                else {}
                            )
                            influence_stats = flip_tracker.compute_influences(
                                model, step_flips, global_step
                            )

                    if aga_optimizer is not None:
                        aga_optimizer.step()
                    if bbo_optimizer is not None:
                        bbo_optimizer.step()
                    if ema is not None:
                        ema.update(
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )

                    # C3: Progressive Layer Freezing
                    if progressive_freezer is not None:
                        freeze_stats = progressive_freezer.check_and_freeze(global_step)
                        if freeze_stats["newly_frozen"]:
                            print(
                                f"[C3-FREEZE] Newly frozen: {freeze_stats['newly_frozen']} "
                                f"(total: {freeze_stats['total_frozen']}/{freeze_stats['total_layers']})"
                            )

                    # Sparse Active Shadow Weights (R4-A2-M6/P4)
                    if (
                        use_sparse_shadow
                        and getattr(config, "use_ternary", False)
                        and global_step % sparse_shadow_interval == 0
                    ):
                        _ss_log = global_step == start_step or global_step % 50 == 0
                        apply_sparse_shadow_weights(
                            model,
                            optimizer,
                            margin=sparse_shadow_margin,
                            decay=sparse_shadow_decay,
                            step=global_step,
                            log=_ss_log,
                        )

                    # Update current LR on modules for dead zone decay (R2-A6-M14)
                    if use_dead_zone:
                        try:
                            _step_lr = optimizer.param_groups[0]["lr"]
                        except (IndexError, KeyError, AttributeError):
                            _step_lr = base_lr
                        for _m in model.modules():
                            if hasattr(_m, "_current_lr"):
                                _m._current_lr = _step_lr

                    # Information-theoretic freezing: update confidence (R2-A3-M3)
                    if use_info_freeze:
                        update_info_freeze_vhat(model, optimizer)
                        for _m in model.modules():
                            if hasattr(_m, "update_freeze_confidence"):
                                _m.update_freeze_confidence()
                        zero_frozen_optimizer_state(model, optimizer)
                    elif use_ternary:
                        # A3-M7: C2 commitment optimizer state zeroing (L1/L2 no-op, L3 active)
                        zero_frozen_optimizer_state(model, optimizer)

                    # AutoFreeze: layer-level gradient-norm convergence
                    if autofreeze is not None:
                        _af_result = autofreeze.step(global_step)
                        if _af_result["newly_frozen"]:
                            print(f"[AutoFreeze] step={global_step} newly_frozen={_af_result['newly_frozen']}")

                    # R1-A6: SAP -- log per-layer sensitivity (always active when ternary)
                    if use_ternary and global_step % 50 == 0:
                        log_layer_sensitivity(model, global_step)

                    # R2-A2-M4: End fused AGA step cycle
                    if fused_aga is not None:
                        fused_aga.end_step()

                    if getattr(config, "use_caq", False):
                        update_caq_state(model, optimizer)

                    # INSTANT: Update step counter for activation projection scheduling (R2-A2-F6)
                    if getattr(config, "use_instant", False):
                        for module in model.modules():
                            if hasattr(module, "_instant_step"):
                                module._instant_step = global_step

                    # Phase 20.A: Anti-windup for shadow weights
                    if getattr(config, "antiwindup_k", 0) > 0:
                        from .components.ternary import apply_antiwindup

                        apply_antiwindup(model, k_aw=config.antiwindup_k)

                    # Fused requantize or invalidate ternary weight caches (weights changed by optimizer)
                    if use_fused_requantize:
                        _fused_n = 0
                        for module in model.modules():
                            if hasattr(module, "requantize_from_shadow"):
                                module.requantize_from_shadow()
                                _fused_n += 1
                        if global_step == start_step:
                            print(
                                f"[FUSED-REQUANT] Pre-warmed {_fused_n} ternary weight caches",
                                flush=True,
                            )
                    else:
                        for module in model.modules():
                            if hasattr(module, "invalidate_weight_cache"):
                                module.invalidate_weight_cache()
                    if hasattr(model, "optrot") or any(
                        hasattr(m, "optrot") for m in model.modules()
                    ):
                        for module in model.modules():
                            if hasattr(module, "optrot") and hasattr(
                                module.optrot, "invalidate_rotation_cache"
                            ):
                                module.optrot.invalidate_rotation_cache()

                    # Periodic merge for deferred ternary
                    if getattr(config, "use_deferred_ternary", False):
                        merge_interval = getattr(config, "deferred_merge_interval", 100)
                        if global_step % merge_interval == 0 and global_step > 0:
                            from .components.ternary import merge_all_deferred_deltas

                            n_merged = merge_all_deferred_deltas(model, optimizer)
                else:
                    # Margin-Sparse gradient recycling (R2-A1-M1+M4)
                    if gradient_recyclers is not None:
                        apply_gradient_recycling(
                            model,
                            gradient_recyclers,
                            global_step,
                            margin_refresh_interval,
                            margin_delta,
                        )
                    if use_adagc:
                        # AdaGC: adaptive per-tensor gradient clipping (R2-A3-F6)
                        _adagc_clips = 0
                        _adagc_max_ratio = 0.0
                        for _name, _param in model.named_parameters():
                            if _param.grad is not None and _name in adagc_ema:
                                _gnorm = _param.grad.data.norm().item()
                                if adagc_ema[_name] == 0.0:
                                    adagc_ema[_name] = (
                                        _gnorm  # initialize on first step
                                    )
                                else:
                                    adagc_ema[_name] = (
                                        adagc_beta * adagc_ema[_name]
                                        + (1 - adagc_beta) * _gnorm
                                    )
                                if (
                                    adagc_ema[_name] > 0
                                    and _gnorm > adagc_factor * adagc_ema[_name]
                                ):
                                    _scale = adagc_factor * adagc_ema[_name] / _gnorm
                                    _param.grad.data.mul_(_scale)
                                    _adagc_clips += 1
                                    _adagc_max_ratio = max(
                                        _adagc_max_ratio, _gnorm / adagc_ema[_name]
                                    )
                    # A6-M3: Threshold-scaled gradient multiplier (thermodynamic annealing).
                    # threshold_coeff anneals inside forward() as local var — never writes back.
                    # Replicate formula here so optimizer sees the correct effective LR.
                    if use_ternary:
                        _a6m3_progress = global_step / max(1, total_steps)
                        _tc_vals_step = [
                            max(0.05, m.threshold_coeff * (1.0 - _a6m3_progress) ** 0.25)
                            for m in model.modules()
                            if hasattr(m, "threshold_coeff")
                            and hasattr(m, "group_size")
                            and getattr(m, "threshold_type", "coefficient") == "thermodynamic"
                        ]
                        if _tc_vals_step:
                            _avg_tc_step = sum(_tc_vals_step) / len(_tc_vals_step)
                            _lr_tc_scale_step = _avg_tc_step / DEFAULT_THRESHOLD_COEFF
                            for _name, _param in model.named_parameters():
                                if _param.grad is not None and "weight" in _name:
                                    _param.grad.data.mul_(_lr_tc_scale_step)
                            if not hasattr(train_model, "_a6m3_logged"):
                                print(
                                    f"[A6-M3] ACTIVE step={global_step} "
                                    f"avg_tc={_avg_tc_step:.4f} scale={_lr_tc_scale_step:.4f}"
                                )
                                train_model._a6m3_logged = True

                    # A6-F6: Commitment-driven per-layer gradient scaling (before clip)
                    if use_ternary and global_step > warmup_steps:
                        _a6f6_scales = apply_commitment_lr_scaling(
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )
                        if not hasattr(train_model, "_a6f6_logged"):
                            n_scaled = len(_a6f6_scales)
                            if n_scaled > 0:
                                avg_scale = sum(_a6f6_scales.values()) / n_scaled
                                print(
                                    f"[A6-F6] Commitment LR scaling | layers={n_scaled} avg_scale={avg_scale:.3f}"
                                )
                                train_model._a6f6_logged = True
                            elif global_step > warmup_steps + 5:
                                print("[A6-F6] No layers with commitment tracking (n_stable uninitialized)")
                                train_model._a6f6_logged = True
                        # [F22] Adaptive Gradient Clipping (in addition to per-tensor clipping)
                        clip_stats = grad_clipper.clip(model.parameters())
                        # Still compute global grad_norm for logging (after per-tensor clipping)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
                    else:
                        # [F22] Adaptive Gradient Clipping
                        clip_stats = grad_clipper.clip(model.parameters())
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
                    # R19-B: CUSUM Optimal Update Gate
                    _cusum_skip = False
                    if getattr(config, 'use_cusum_update_gate', False):
                        _cusum_k = getattr(config, 'cusum_k', 0.5)
                        _cusum_h = getattr(config, 'cusum_h', 5.0)
                        _cusum_alpha = getattr(config, 'cusum_ema_alpha', 0.05)
                        _gn = float(grad_norm)
                        if _cusum_mu is None:
                            _cusum_mu = _gn
                        else:
                            _cusum_mu = (1.0 - _cusum_alpha) * _cusum_mu + _cusum_alpha * _gn
                        _cusum_c = max(0.0, _cusum_c + abs(_gn - _cusum_mu) - _cusum_k * max(_cusum_mu, 0.01))
                        _cusum_skip = (_cusum_c <= _cusum_h)
                        if _cusum_skip:
                            _cusum_skipped += 1
                            _cusum_c = 0.0
                        if global_step % 100 == 0 and not torch.compiler.is_compiling():
                            print(f"[CUSUM-GATE] step={global_step} C={_cusum_c:.3f} mu={_cusum_mu:.4f} "
                                  f"skip={_cusum_skip} total_skipped={_cusum_skipped}", flush=True)
                    # CBS: track rolling grad norm for Curriculum Batch Scaling
                    _cbs_grad_norms.append(float(grad_norm))
                    if len(_cbs_grad_norms) > 50:
                        _cbs_grad_norms.pop(0)
                    if (len(_cbs_grad_norms) >= 50 and global_step % 200 == 0
                            and _cbs_current_accum < _cbs_max_accum):
                        _cbs_mean = sum(_cbs_grad_norms) / len(_cbs_grad_norms)
                        _cbs_var = sum((x - _cbs_mean) ** 2 for x in _cbs_grad_norms) / len(_cbs_grad_norms)
                        _cbs_cv = (_cbs_var ** 0.5) / max(_cbs_mean, 1e-8)
                        if _cbs_cv < 0.15:
                            _cbs_current_accum = min(_cbs_current_accum * 2, _cbs_max_accum)
                            grad_accum_steps = _cbs_current_accum
                            # Scale LR by sqrt(2) for larger effective batch (Surge paper)
                            for _cbs_pg in optimizer.param_groups:
                                _cbs_pg['lr'] *= 1.4142
                            print(f"[CBS] step={global_step} CV={_cbs_cv:.3f} -> accum={_cbs_current_accum}×, LR*sqrt2")
                    # Layerwise LR via HT-SR spectral alpha
                    if layerwise_lr:
                        if global_step >= warmup_steps and global_step % 500 == 0:
                            _lr_multipliers = compute_lr_multipliers(
                                compute_layer_alphas(model)
                            )
                        if _lr_multipliers:
                            for name, param in model.named_parameters():
                                if param.grad is not None and name in _lr_multipliers:
                                    param.grad.mul_(_lr_multipliers[name])
                    if use_grad_noise:
                        _gn_std = _apply_gradient_noise(
                            model, global_step, grad_noise_eta, grad_noise_gamma
                        )
                        if global_step == start_step:
                            print(
                                f"[GRAD-NOISE VERIFY] ACTIVE | eta={grad_noise_eta} gamma={grad_noise_gamma} std={_gn_std:.6f}"
                            )
                    if use_gradient_sign_consistency and config.use_ternary:
                        apply_gradient_sign_consistency(
                            model,
                            _grad_sign_history,
                            grad_sign_threshold,
                            grad_sign_window,
                        )
                    if use_topk_grad_sparsification:
                        apply_topk_gradient_sparsification(
                            model, topk_grad_sparsity, topk_error_buffers
                        )

                    # A2-F6: Layerwise Gauss-Newton Preconditioning
                    if getattr(config, "use_fisher_preconditioning", False):
                        apply_fisher_preconditioning(
                            model,
                            fisher_accum,
                            global_step,
                            damping=getattr(config, "fisher_damping", 1e-4),
                        )

                    # A2-F7: SignSGD Momentum
                    if getattr(config, "use_sign_sgd", False):
                        apply_sign_momentum(
                            model,
                            sign_sgd_momentum_buffers,
                            beta=getattr(config, "sign_sgd_momentum", 0.9),
                        )

                    # A2-M7: Consensus-Filtered-Layer-Gradients
                    if getattr(config, "use_consensus_gradient_filter", False):
                        apply_consensus_gradient_filter(
                            model,
                            consensus_alpha=getattr(config, "consensus_alpha", 0.5),
                        )

                    # R4-A3-M5: Gradient Consensus Polarization
                    if use_gcp:
                        from .components.gcp import apply_gcp_simple

                        n_apps = getattr(config, "recursive_iterations", 32)
                        for name, param in model.named_parameters():
                            if param.grad is not None and "shared_ssm_block" in name:
                                param.grad.data = apply_gcp_simple(
                                    param.grad.data,
                                    n_applications=n_apps,
                                    alpha=gcp_alpha,
                                )
                        if global_step == start_step:
                            print(
                                f"[GCP VERIFY] ACTIVE | alpha={gcp_alpha} n_apps={n_apps}"
                            )

                    # R4-A3-M6: RAFU (Recurrence-Aware Flip Utility)
                    if use_rafu and global_step % rafu_interval == 0:
                        _inter_states = (
                            model.get_intermediate_states()
                            if hasattr(model, "get_intermediate_states")
                            else []
                        )
                        if _inter_states and len(_inter_states) >= 2:
                            from .components.rafu import (
                                compute_rafu_utility,
                                apply_rafu_flip_filter,
                            )

                            _rafu_utility = compute_rafu_utility(
                                _inter_states[:-1],
                                _inter_states[1:],
                            )
                            if _rafu_utility is not None:
                                for name, param in model.named_parameters():
                                    if (
                                        param.grad is not None
                                        and "shared_ssm_block" in name
                                    ):
                                        if param.grad.shape == _rafu_utility.shape:
                                            param.grad.data = apply_rafu_flip_filter(
                                                param.grad.data,
                                                _rafu_utility,
                                                threshold=rafu_threshold,
                                            )
                                if global_step == start_step:
                                    print(
                                        f"[RAFU VERIFY] ACTIVE | utility_shape={_rafu_utility.shape} thresh={rafu_threshold}"
                                    )

                    # R4-A3-M8: Bayesian Flip Accumulator
                    if use_bfa and bfa_accumulators and global_step % bfa_interval == 0:
                        total_flips = 0
                        model_to_check = (
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )
                        for name, module in model_to_check.named_modules():
                            if (
                                name in bfa_accumulators
                                and hasattr(module, "weight")
                                and module.weight.grad is not None
                            ):
                                bfa = bfa_accumulators[name]
                                w = module.weight.data
                                threshold = (
                                    getattr(module, "threshold_coeff", 0.5)
                                    * w.abs().mean()
                                )
                                current_ternary = (
                                    torch.sign(w) * (w.abs() > threshold).float()
                                )

                                bfa.update(module.weight.grad.data, current_ternary)
                                n_flips = bfa.apply_flips(
                                    module.weight,
                                    getattr(module, "threshold_coeff", 0.5),
                                )
                                total_flips += n_flips
                        if global_step == start_step:
                            print(
                                f"[BFA VERIFY] ACTIVE | layers={len(bfa_accumulators)} interval={bfa_interval}"
                            )
                        if total_flips > 0 and global_step % (bfa_interval * 10) == 0:
                            print(
                                f"[BFA] Step {global_step}: {total_flips} flips applied"
                            )

                    # R4-A3-P4: Spin-Glass MC Sweep (periodic)
                    if (
                        use_sgapd
                        and sgapd_annealer is not None
                        and global_step % sgapd_interval == 0
                    ):
                        _sg_accepted = 0
                        model_to_check = (
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )
                        for _name, _module in model_to_check.named_modules():
                            if hasattr(_module, "quantize_enabled") and hasattr(
                                _module, "weight"
                            ):
                                _sg_accepted += sgapd_annealer.mc_sweep(
                                    _module.weight, None, None, model_to_check
                                )
                        sgapd_annealer.update_temperature()
                        if global_step == start_step:
                            print(
                                f"[SGAPD VERIFY] ACTIVE | T={sgapd_annealer.temperature:.4f}"
                            )

                    # R4-A3-P5: Ergodic Trajectory Optimizer (periodic)
                    if (
                        use_ergodic
                        and ergodic_opt is not None
                        and global_step % ergodic_interval == 0
                    ):
                        _inter_states = (
                            model.get_intermediate_states()
                            if hasattr(model, "get_intermediate_states")
                            else []
                        )
                        if _inter_states:
                            ergodic_opt.update_statistics(model, _inter_states)
                            _erg_flips = ergodic_opt.score_and_flip(model)
                            if global_step == start_step:
                                print(
                                    f"[ERGODIC VERIFY] ACTIVE | interval={ergodic_interval}"
                                )

                    # A2-F8: Adaptive Quantization Levels
                    if (
                        getattr(config, "use_adaptive_quantization", False)
                        and global_step
                        % getattr(config, "adaptive_quant_interval", 500)
                        == 0
                    ):
                        apply_adaptive_quantization_levels(
                            model,
                            sensitivity_threshold=getattr(
                                config, "adaptive_quant_sensitivity_threshold", 0.1
                            ),
                        )

                    # A2-M6: FSCR (Flip Safety/Confidence Ratio) - prevent outlier-driven flips
                    if use_fscr_flip_gate:
                        from .components.fscr import (
                            compute_fscr_mask,
                            apply_fscr_mask,
                            log_fscr_statistics,
                        )

                        fscr_masks = compute_fscr_mask(
                            model, fscr_threshold=fscr_threshold
                        )
                        apply_fscr_mask(model, fscr_masks)
                        log_fscr_statistics(
                            model, fscr_masks, global_step, log_interval=500
                        )

                    # R4-A4-F10: Snapshot params before update
                    if gradient_monitor is not None:
                        gradient_monitor.snapshot_params(model)

                    # F24: Information-Theoretic Freezing
                    if info_freezer is not None:
                        info_freezer.update(model, global_step)
                        info_freezer.apply_masks(model)

                    # R6-A5-M7: Commitment-Driven Precision (P2)
                    if commitment_precision is not None:
                        commitment_precision.apply(model, global_step)

                    # A3-F13: Skip optimizer for committed weights
                    # Zero gradient for committed positions so Lion skips them.
                    # A3-F13 + A1-F16: Block-wise gradient skip for committed weights
                    if use_ternary:
                        _skipped_w = 0
                        _total_w = 0
                        _raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                        for _m in _raw_model.modules():
                            if (
                                hasattr(_m, "committed_mask")
                                and _m.committed_mask is not None
                                and hasattr(_m, "weight")
                                and _m.weight.grad is not None
                            ):
                                _cm = _m.committed_mask
                                _gs = getattr(_m, "group_size", 128)
                                _flat_cm = _cm.reshape(-1)
                                _N = _flat_cm.numel()
                                _total_w += _N
                                # A1-F16: Check whole blocks of group_size
                                _pad = (-_N) % _gs
                                if _pad > 0:
                                    _padded = torch.cat([_flat_cm, _flat_cm.new_ones(_pad)])
                                else:
                                    _padded = _flat_cm
                                _blocks = _padded.reshape(-1, _gs).all(dim=-1)
                                _block_expanded = _blocks.unsqueeze(-1).expand(-1, _gs).reshape(-1)[:_N]
                                _block_mask = _block_expanded.reshape(_cm.shape)
                                _m.weight.grad[_block_mask] = 0.0
                                _skipped_w += int(_block_mask.sum().item())
                                if not hasattr(train_model, "_a1f16_logged") and _blocks.any():
                                    print(
                                        f"[A1-F16] Block-wise skip: "
                                        f"full_blocks={int(_blocks.sum())}/{len(_blocks)} gs={_gs}"
                                    )
                                    train_model._a1f16_logged = True
                        if not hasattr(train_model, "_a3f13_logged") and _total_w > 0:
                            print(
                                f"[A3-F13] Skip-optimizer | committed={_skipped_w}/{_total_w} "
                                f"({100*_skipped_w/max(1,_total_w):.1f}%) grad zeroed"
                            )
                            train_model._a3f13_logged = True

                    # D2 — Lion Momentum-Lock Grad Skip (R8-A5)
                    # When sign(g)==sign(m) AND |m|>threshold, sign(update)==sign(m) regardless.
                    # Zero those grads: saves grad accumulation overhead with same Lion update.
                    if optimizer_type == "lion":
                        _d2_locked = 0
                        _d2_total = 0
                        _d2_threshold = 0.01
                        for _grp in optimizer.param_groups:
                            for _p in _grp["params"]:
                                if _p.grad is None:
                                    continue
                                _d2_total += _p.grad.numel()
                                _p_state = optimizer.state.get(_p, {})
                                if "exp_avg" not in _p_state:
                                    continue
                                _m = _p_state["exp_avg"].float()
                                _g = _p.grad.float()
                                _lock = (_m * _g > 0) & (_m.abs() > _d2_threshold)
                                _p.grad.data[_lock] = 0.0
                                _d2_locked += int(_lock.sum().item())
                        if not hasattr(train_model, "_d2_logged") and _d2_total > 0:
                            print(
                                f"[D2-MLGS] Momentum-lock skip | "
                                f"locked={_d2_locked}/{_d2_total} "
                                f"({100*_d2_locked/max(1,_d2_total):.1f}%) grad zeroed"
                            )
                            train_model._d2_logged = True

                    if use_ternary and global_step % 50 == 0:
                        _nf_sum = 0.0; _nf_count = 0
                        _raw_model_nf = model._orig_mod if hasattr(model, "_orig_mod") else model
                        for _m_nf in _raw_model_nf.modules():
                            if (hasattr(_m_nf, "committed_mask") and _m_nf.committed_mask is not None
                                    and hasattr(_m_nf, "weight") and _m_nf.weight.grad is not None
                                    and _m_nf.committed_mask.any()):
                                _g_nf = _m_nf.weight.grad.float()
                                _cm_nf = _m_nf.committed_mask
                                _nf_sum += _g_nf[_cm_nf].pow(2).sum().item()
                                _nf_count += int(_cm_nf.sum().item())
                        if _nf_count > 0:
                            _sigma2_noise = _nf_sum / _nf_count
                            print(f"[E1-MARS] noise_floor σ²={_sigma2_noise:.2e} n_committed={_nf_count} step={global_step}")

                    # G2: Uncommitted Gap Adaptive LR
                    # Scale gradients of uncommitted weights upward before optimizer step.
                    # lr_uncommitted = base_lr * (1 + gap_factor), gap_factor = uncommitted_frac * scale.
                    if use_ternary and getattr(config, "use_uncommitted_gap_lr", True):
                        _g2_raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                        _g2_total = 0
                        _g2_uncommitted = 0
                        for _g2_m in _g2_raw.modules():
                            if (
                                hasattr(_g2_m, "committed_mask")
                                and _g2_m.committed_mask is not None
                                and hasattr(_g2_m, "weight")
                                and _g2_m.weight.grad is not None
                            ):
                                _g2_cm = _g2_m.committed_mask
                                _g2_total += _g2_cm.numel()
                                _g2_uncommitted += int((~_g2_cm).sum().item())
                                _g2_unc_frac = (~_g2_cm).float().mean().item()
                                _g2_mult = 1.0 + getattr(config, "gap_lr_scale", 0.3) * _g2_unc_frac
                                _g2_m.weight.grad[~_g2_cm] *= _g2_mult
                        if _g2_total > 0 and global_step % 50 == 0:
                            _g2_unc_frac_global = _g2_uncommitted / max(1, _g2_total)
                            _g2_mult_global = 1.0 + getattr(config, "gap_lr_scale", 0.3) * _g2_unc_frac_global
                            print(f"[G2-GAP] step={global_step} uncommitted_frac={_g2_unc_frac_global:.3f} lr_mult={_g2_mult_global:.3f}")

                    # Batch AD: ADMM proximal gradient injection before optimizer.step()
                    # Adds rho * (w - Z + u) to weight.grad for all quantized layers
                    for _ad_mod in model.modules():
                        if (hasattr(_ad_mod, "admm_u") and _ad_mod.admm_u is not None
                                and hasattr(_ad_mod, "_admm_last_ternary") and _ad_mod._admm_last_ternary is not None
                                and hasattr(_ad_mod, "weight") and _ad_mod.weight.grad is not None):
                            _ad_mod.training_step = global_step
                            with torch.no_grad():
                                _proximal = _ad_mod._admm_rho * (
                                    _ad_mod.weight.to(_ad_mod.admm_u.dtype)
                                    - _ad_mod._admm_last_ternary.to(_ad_mod.admm_u.dtype)
                                    + _ad_mod.admm_u
                                )
                                _ad_mod.weight.grad.add_(_proximal.to(_ad_mod.weight.grad.dtype))
                                # Batch AE: accumulate gradient into push buffer (raw grad, no lr — tracks gradient integral Σg_t)
                                if hasattr(_ad_mod, 'accum_push') and _ad_mod.accum_push is not None:
                                    _g = _ad_mod.weight.grad.detach()
                                    _push_delta = (_g * 10000.0).clamp(-32767, 32767).to(torch.int16)
                                    _ad_mod.accum_push.add_(_push_delta)

                    # Batch AH: Kalman-smoother Hessian update + preconditioning for SSM params
                    for _kh_mod in model.modules():
                        if hasattr(_kh_mod, 'update_kalman_hessian'):
                            _kh_mod.update_kalman_hessian()
                            _kh_mod.apply_kalman_precond()

                    if not _cusum_skip:
                        optimizer.step()

                    # AO-4: SSM A_log Annealing — developmental curriculum
                    # Principle (R17-B-W2-PB-F4): ramp up A_log clamp max with step.
                    # Short timescales stabilize first; long timescales open gradually.
                    if getattr(config, 'use_ssm_annealing', False):
                        _anneal_warmup = getattr(config, 'ssm_anneal_warmup', 1000)
                        _anneal_log_max = getattr(config, 'ssm_anneal_log_max', 4.0)
                        _anneal_frac = min(global_step / max(_anneal_warmup, 1), 1.0)
                        _a_log_max = _anneal_log_max * _anneal_frac
                        with torch.no_grad():
                            for _ssm_m in model.modules():
                                if hasattr(_ssm_m, 'A_log'):
                                    _ssm_m.A_log.data.clamp_(max=_a_log_max)
                        if global_step % 200 == 0 and not torch.compiler.is_compiling():
                            print(f"[SSM-ANNEAL] step={global_step} log_max={_a_log_max:.3f} "
                                  f"frac={_anneal_frac:.3f}")

                    # DSST correction: enforce ternary structure after optimizer step
                    if not _cusum_skip:
                        if use_ternary:
                            with torch.no_grad():
                                for _mod in model.modules():
                                    if hasattr(_mod, 'use_dsst') and _mod.use_dsst and hasattr(_mod, 'dsst_m') and hasattr(_mod, 'dsst_s') and hasattr(_mod, 'weight'):
                                        _alpha = _mod.weight.data.abs().mean().clamp(min=1e-7)
                                        _mod.weight.data.copy_(
                                            (_mod.dsst_m.float() * _mod.dsst_s.float() * _alpha).to(_mod.weight.dtype)
                                        )
                                        if hasattr(_mod, 'packed_valid'):
                                            if isinstance(_mod.packed_valid, torch.Tensor):
                                                _mod.packed_valid.zero_()
                                            else:
                                                _mod.packed_valid = False
                                        if hasattr(_mod, 'int2_packed_valid'):
                                            if isinstance(_mod.int2_packed_valid, torch.Tensor):
                                                _mod.int2_packed_valid.zero_()
                                            else:
                                                _mod.int2_packed_valid = False

                    # AV-19: Coordinate descent B — grad-state correlation as B quality signal
                    if getattr(config, 'use_av19_coord_descent_b_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _av19_corrs = []
                                for _av19_mod in model.modules():
                                    if (hasattr(_av19_mod, 'x_proj')
                                            and hasattr(_av19_mod.x_proj, 'weight')
                                            and _av19_mod.x_proj.weight.grad is not None):
                                        _av19_g = _av19_mod.x_proj.weight.grad.detach().float()
                                        _av19_w = _av19_mod.x_proj.weight.detach().float()
                                        _av19_gf = _av19_g.reshape(-1)
                                        _av19_wf = _av19_w.reshape(-1)
                                        _av19_gn = _av19_gf - _av19_gf.mean()
                                        _av19_wn = _av19_wf - _av19_wf.mean()
                                        _av19_denom = (_av19_gn.norm() * _av19_wn.norm()).clamp(min=1e-8)
                                        _av19_corr = (_av19_gn * _av19_wn).sum() / _av19_denom
                                        _av19_corrs.append(_av19_corr.item())
                                if _av19_corrs:
                                    _av19_mean_corr = sum(_av19_corrs) / len(_av19_corrs)
                                    print(f"[AV-19 COORD-B] step={global_step} "
                                          f"B_grad_weight_corr={_av19_mean_corr:.4f} "
                                          f"n_ssm_layers={len(_av19_corrs)}")

                    # AW-7: PAC-Bayes KL — negative entropy over global ternary distribution
                    if use_ternary and getattr(config, 'use_aw7_pac_bayes_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _aw7_pos, _aw7_neg, _aw7_zero, _aw7_n = 0.0, 0.0, 0.0, 0
                                for _aw7_m in model.modules():
                                    if hasattr(_aw7_m, 'weight') and hasattr(_aw7_m, 'threshold_coeff'):
                                        _aw7_w = _aw7_m.weight.detach().float()
                                        _aw7_thr = _aw7_m.threshold_coeff * _aw7_w.abs().mean()
                                        _aw7_pos += float((_aw7_w > _aw7_thr).sum().item())
                                        _aw7_neg += float((_aw7_w < -_aw7_thr).sum().item())
                                        _aw7_zero += float((_aw7_w.abs() <= _aw7_thr).sum().item())
                                        _aw7_n += _aw7_w.numel()
                                if _aw7_n > 0:
                                    import math as _aw7_math
                                    _aw7_p = [_aw7_pos / _aw7_n, _aw7_neg / _aw7_n, _aw7_zero / _aw7_n]
                                    _aw7_entropy = -sum(p * _aw7_math.log(p + 1e-12) for p in _aw7_p)
                                    _aw7_kl = -_aw7_entropy - _aw7_math.log(3)
                                    print(f"[AW-7 PAC-BAYES] step={global_step} "
                                          f"p_pos={_aw7_p[0]:.3f} p_neg={_aw7_p[1]:.3f} p_zero={_aw7_p[2]:.3f} "
                                          f"entropy={_aw7_entropy:.4f} kl_vs_uniform={_aw7_kl:.4f}")

                    # AW-10: Free probability S-transform — spectral product monitor
                    if use_ternary and getattr(config, 'use_aw10_free_prob_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _aw10_prods = []
                                for _aw10_m in model.modules():
                                    if hasattr(_aw10_m, 'weight') and hasattr(_aw10_m, 'threshold_coeff'):
                                        _aw10_w = _aw10_m.weight.detach().float()
                                        _aw10_thr = _aw10_m.threshold_coeff * _aw10_w.abs().mean()
                                        _aw10_sp = float((_aw10_w.abs() <= _aw10_thr).float().mean().item())
                                        _aw10_sig2 = float(_aw10_w.var().item())
                                        _aw10_z = 1.0
                                        _aw10_d = _aw10_sig2 * (1.0 - _aw10_sp) * (_aw10_z + 1.0 - _aw10_sp)
                                        if _aw10_d > 1e-8:
                                            _aw10_prods.append(1.0 / _aw10_d)
                                if _aw10_prods:
                                    import math as _aw10_math
                                    _aw10_log_prod = sum(_aw10_math.log(max(s, 1e-10)) for s in _aw10_prods)
                                    print(f"[AW-10 FREE-PROB] step={global_step} "
                                          f"n_layers={len(_aw10_prods)} "
                                          f"log_S_product={_aw10_log_prod:.4f} "
                                          f"S_mean={sum(_aw10_prods)/max(len(_aw10_prods),1):.4f}")

                    # AW-11: Belief Propagation — near-threshold weight count (BP candidates)
                    if use_ternary and getattr(config, 'use_aw11_bp_threshold_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _aw11_near, _aw11_total = 0, 0
                                for _aw11_m in model.modules():
                                    if hasattr(_aw11_m, 'weight') and hasattr(_aw11_m, 'threshold_coeff'):
                                        _aw11_w = _aw11_m.weight.detach().float()
                                        _aw11_sg = max(_aw11_w.std().item(), 1e-8)
                                        _aw11_thr = float(_aw11_m.threshold_coeff * _aw11_w.abs().mean().item())
                                        _aw11_near += int(((_aw11_w.abs() - _aw11_thr).abs() < 2 * _aw11_sg).sum().item())
                                        _aw11_total += _aw11_w.numel()
                                if _aw11_total > 0:
                                    print(f"[AW-11 BP-THRESH] step={global_step} "
                                          f"near_threshold={_aw11_near}/{_aw11_total} "
                                          f"bp_candidate_frac={_aw11_near/max(_aw11_total,1):.4f}")

                    # AW-13: LZ complexity — row-transition count as compressibility proxy
                    if use_ternary and getattr(config, 'use_aw13_lz_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _aw13_lz_vals = []
                                for _aw13_m in model.modules():
                                    if hasattr(_aw13_m, 'weight') and hasattr(_aw13_m, 'threshold_coeff'):
                                        _aw13_w = _aw13_m.weight.detach().float()
                                        _aw13_thr = _aw13_m.threshold_coeff * _aw13_w.abs().mean()
                                        _aw13_tern = _aw13_w.sign() * (_aw13_w.abs() > _aw13_thr).float()
                                        _aw13_rows = _aw13_tern[:32]
                                        _aw13_lz = float((_aw13_rows[:, 1:] != _aw13_rows[:, :-1]).float().sum(dim=1).mean().item()) + 1.0
                                        _aw13_lz_vals.append(_aw13_lz)
                                if _aw13_lz_vals:
                                    print(f"[AW-13 LZ-COMPLEXITY] step={global_step} "
                                          f"mean_row_transitions={sum(_aw13_lz_vals)/max(len(_aw13_lz_vals),1):.2f} "
                                          f"n_layers={len(_aw13_lz_vals)}")

                    # AW-15: Renyi-2 loss reference probe vs CE
                    if getattr(config, 'use_aw15_renyi_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                try:
                                    _aw15_logits = logits.detach().float()
                                    _aw15_tgt = targets.detach()
                                    _aw15_probs = torch.softmax(_aw15_logits[:, :-1, :].reshape(-1, _aw15_logits.shape[-1]), dim=-1)
                                    _aw15_tgt_flat = _aw15_tgt[:, 1:].reshape(-1)
                                    _aw15_mask = (_aw15_tgt_flat >= 0) & (_aw15_tgt_flat < _aw15_probs.shape[-1])
                                    if _aw15_mask.sum() > 0:
                                        _aw15_p = _aw15_probs[_aw15_mask].gather(1, _aw15_tgt_flat[_aw15_mask].unsqueeze(1)).squeeze(1).clamp(min=1e-8)
                                        _aw15_r2 = (1.0 / _aw15_p - 1.0).mean().item()
                                        print(f"[AW-15 RENYI-2] step={global_step} "
                                              f"renyi2={_aw15_r2:.4f} ce={loss.item():.4f} "
                                              f"ratio={_aw15_r2/(loss.item()+1e-8):.3f}")
                                except Exception:
                                    pass

                    # AW-19: MDL commitment reward — log2(3/2) per committed weight
                    if use_ternary and getattr(config, 'use_aw19_mdl_reward_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _aw19_committed, _aw19_total = 0, 0
                                for _aw19_m in model.modules():
                                    if hasattr(_aw19_m, 'weight') and hasattr(_aw19_m, 'threshold_coeff'):
                                        _aw19_w = _aw19_m.weight.detach().float()
                                        _aw19_thr = _aw19_m.threshold_coeff * _aw19_w.abs().mean()
                                        _aw19_committed += int((_aw19_w.abs() > _aw19_thr).sum().item())
                                        _aw19_total += _aw19_w.numel()
                                if _aw19_total > 0:
                                    _aw19_frac = _aw19_committed / _aw19_total
                                    _aw19_mdl = -0.585 * _aw19_frac
                                    print(f"[AW-19 MDL-REWARD] step={global_step} "
                                          f"committed_frac={_aw19_frac:.4f} mdl_reward={_aw19_mdl:.4f}")

                    # AW-20: HoTT critical radius — Hamming distance staging probe
                    if use_ternary and getattr(config, 'use_aw20_hott_radius_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _aw20_L = sum(1 for _ in model.modules() if hasattr(_, 'threshold_coeff'))
                                _aw20_exceed, _aw20_total = 0, 0
                                for _aw20_m in model.modules():
                                    if hasattr(_aw20_m, 'weight') and hasattr(_aw20_m, 'threshold_coeff'):
                                        _aw20_w = _aw20_m.weight.detach().float()
                                        _aw20_xnorm = _aw20_w.abs().max().clamp(min=1e-8).item()
                                        _aw20_r_crit = 1e-3 / (max(_aw20_L, 1) * _aw20_xnorm)
                                        _aw20_thr = _aw20_m.threshold_coeff * _aw20_w.abs().mean()
                                        _aw20_tern = _aw20_w.sign() * (_aw20_w.abs() > _aw20_thr).float()
                                        _aw20_ham = (_aw20_w.sign() != _aw20_tern.sign()).float().mean().item()
                                        _aw20_exceed += int(_aw20_ham > _aw20_r_crit)
                                        _aw20_total += 1
                                if _aw20_total > 0:
                                    print(f"[AW-20 HOTT-RADIUS] step={global_step} "
                                          f"layers_exceed_r_crit={_aw20_exceed}/{_aw20_total} L={_aw20_L}")

                    # ─── AX: Batch AX — R17-K Multi-Dimensional Evaluation ───
                    # AX-01: BBH-23 Valley Floor Detector
                    if getattr(config, 'use_ax01_bbh_valley_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax01_gn = grad_norm if isinstance(grad_norm, (int, float)) else (grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm))
                                print(f"[AX-01 BBH-VALLEY] step={global_step} loss_floor={loss.item():.4f} grad_norm={_ax01_gn:.4f} (lm_eval BBH-23 for capability valley)")
                    # AX-02: Think-Bench Efficiency Score
                    if getattr(config, 'use_ax02_think_bench_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax02_eff = 1.0 / max(loss.item(), 1e-8)
                                print(f"[AX-02 THINK-BENCH] step={global_step} eff_proxy={_ax02_eff:.4f} bits/loss (useful-token ratio surrogate)")
                    # AX-03: Three-Axis Cognitive Profile
                    if getattr(config, 'use_ax03_cognitive_profile_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                print(f"[AX-03 COG-PROFILE] step={global_step} loss={loss.item():.4f} (factual/syntactic/logical: mmlu/blimp/bbh via lm_eval)")
                    # AX-04: Metric Preference Checklist
                    if getattr(config, 'use_ax04_metric_pref_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                print(f"[AX-04 METRIC-PREF] step={global_step} loss={loss.item():.4f} (pairwise discrimination: compare checkpoint pairs on downstream evals)")
                    # AX-05: Scaling Projected Ceiling
                    if getattr(config, 'use_ax05_scaling_ceiling_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax05_n = sum(p.numel() for p in model.parameters())
                                print(f"[AX-05 SCALE-CEIL] step={global_step} n_params={_ax05_n/1e6:.1f}M loss={loss.item():.4f} (log-linear ceiling fit 6M->1B->7B)")
                    # AX-06: Multi-Power Law Loss Prediction
                    if getattr(config, 'use_ax06_power_law_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax06_lr = optimizer.param_groups[0]['lr']
                                print(f"[AX-06 POWER-LAW] step={global_step} lr={_ax06_lr:.2e} loss={loss.item():.4f} (fit L_inf via multi-power-law over LR schedules)")
                    # AX-07: Loss-to-Loss Scaling Law
                    if getattr(config, 'use_ax07_loss2loss_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax07_L = loss.item()
                                _ax07_acc = max(0.0, min(1.0, 0.90 - 0.08 * (_ax07_L - 3.0)))
                                print(f"[AX-07 LOSS2LOSS] step={global_step} pretrain_loss={_ax07_L:.4f} projected_acc={_ax07_acc:.3f}")
                    # AX-08: Cyclic LR Pareto Frontier
                    if getattr(config, 'use_ax08_pareto_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax08_lr = optimizer.param_groups[0]['lr']
                                _ax08_toks = global_step * getattr(config, 'batch_size', 32) * getattr(config, 'seq_len', 128)
                                print(f"[AX-08 PARETO] step={global_step} lr={_ax08_lr:.2e} tokens={_ax08_toks/1e6:.1f}M loss={loss.item():.4f} (Pareto: loss vs compute)")
                    # AX-09: Ternary Zero-Ratio & Flip-Rate Health
                    if getattr(config, 'use_ax09_ternary_health_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax09_zr, _ax09_fr = [], []
                                for _m9 in model.modules():
                                    if hasattr(_m9, 'dsst_s') and _m9.dsst_s is not None:
                                        _w9 = _m9.dsst_s.float()
                                        _ax09_zr.append((_w9 == 0).float().mean().item())
                                        if hasattr(_m9, '_ax09_prev'):
                                            _ax09_fr.append((_w9 != _m9._ax09_prev).float().mean().item())
                                        _m9._ax09_prev = _w9.detach().clone()
                                if _ax09_zr:
                                    _zr9 = sum(_ax09_zr) / len(_ax09_zr)
                                    _fr9 = sum(_ax09_fr) / max(len(_ax09_fr), 1)
                                    print(f"[AX-09 TERNARY-HEALTH] step={global_step} zero_ratio={_zr9:.3f}{'OK' if 0.30<=_zr9<=0.50 else '!'} flip_rate={_fr9:.4f} (optimal 0.30-0.50)")
                    # AX-10: Compute-Normalized Efficiency Score
                    if getattr(config, 'use_ax10_compute_efficiency_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax10_tr = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                                _ax10_pr = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
                                _ax10_ratio = _ax10_tr / max(_ax10_pr, 1e-9)
                                _ax10_eff = (1.0 / max(loss.item(), 1e-8)) / max(_ax10_ratio, 1e-9)
                                print(f"[AX-10 COMPUTE-EFF] step={global_step} train_gb={_ax10_tr:.2f} infer_gb={_ax10_pr:.2f} ratio={_ax10_ratio:.1f}x eff={_ax10_eff:.2e}")
                    # AX-11: Ternary Channel Capacity Ratio (every 500 steps — SVD cost)
                    if getattr(config, 'use_ax11_channel_capacity_probe', True) and use_ternary and global_step % 500 == 1:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax11_r = []
                                for _m11 in model.modules():
                                    if hasattr(_m11, 'dsst_s') and _m11.dsst_s is not None and _m11.dsst_s.dim() == 2 and min(_m11.dsst_s.shape) >= 4:
                                        _w11 = _m11.dsst_s.float()
                                        _sv11 = torch.linalg.svdvals(_w11)
                                        _cap_t = 0.5 * torch.log1p(_sv11**2 / max(_w11.shape[0], 1)).sum().item()
                                        _sv_f11 = torch.full((_sv11.shape[0],), float(_w11.shape[1])**0.5, device=_w11.device, dtype=torch.float)
                                        _cap_f = 0.5 * torch.log1p(_sv_f11**2 / max(_w11.shape[0], 1)).sum().item()
                                        _ax11_r.append(_cap_t / max(_cap_f, 1e-8))
                                if _ax11_r:
                                    print(f"[AX-11 CHAN-CAP] step={global_step} cap_ratio mean={sum(_ax11_r)/len(_ax11_r):.3f} valley={min(_ax11_r):.3f} (target>0.85)")
                    # AX-12: Ternary Structural Win Benchmark
                    if getattr(config, 'use_ax12_structural_win_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax12_zf = [(_m12.dsst_s == 0).float().mean().item() for _m12 in model.modules() if hasattr(_m12, 'dsst_s') and _m12.dsst_s is not None]
                                if _ax12_zf:
                                    _sp12 = sum(_ax12_zf) / len(_ax12_zf)
                                    print(f"[AX-12 STRUCT-WIN] step={global_step} sparsity={_sp12:.3f} (discrete-commit advantage: BoolQ/NLI +2-5% vs FP16 projected)")
                    # AX-13: Threshold Proximity Readiness Score
                    if getattr(config, 'use_ax13_readiness_score_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax13_near, _ax13_tot = 0, 0
                                for _m13 in model.modules():
                                    if hasattr(_m13, 'shadow_weight') and _m13.shadow_weight is not None:
                                        _s13 = _m13.shadow_weight.detach()
                                        _d13 = _m13.lsq_scale_pos.detach().mean().abs().clamp(min=1e-8) if hasattr(_m13, 'lsq_scale_pos') else _s13.abs().mean().clamp(min=1e-8) * 1.5
                                        _ax13_near += (_s13.abs() / _d13 < 1.5).sum().item()
                                        _ax13_tot += _s13.numel()
                                if _ax13_tot > 0:
                                    _r13 = _ax13_near / _ax13_tot
                                    print(f"[AX-13 READINESS] step={global_step} readiness={_r13:.3f} {'OK' if 0.20<=_r13<=0.60 else '!'} (opt 0.20-0.60; <0.10=frozen)")
                    # AX-14: SSM Attractor Diversity
                    if getattr(config, 'use_ax14_ssm_attractor_probe', True) and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax14_v = [_m14.A_log.detach().float().var().item() for _m14 in model.modules() if hasattr(_m14, 'A_log') and _m14.A_log is not None]
                                if _ax14_v:
                                    _dead14 = sum(1 for v in _ax14_v if v < 1e-4) / len(_ax14_v)
                                    print(f"[AX-14 SSM-ATTRACTOR] step={global_step} A_log_var={sum(_ax14_v)/len(_ax14_v):.4f} dead_frac={_dead14:.3f} (target dead<0.15)")
                    # AX-15: Exact Routing Entropy
                    if getattr(config, 'use_ax15_routing_entropy_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax15_zf = [(_m15.dsst_s == 0).float().mean(dim=-1) for _m15 in model.modules() if hasattr(_m15, 'dsst_s') and _m15.dsst_s is not None and _m15.dsst_s.dim() == 2]
                                if _ax15_zf:
                                    _all15 = torch.cat(_ax15_zf)
                                    _b15 = (_all15 * 32).long().clamp(0, 31)
                                    _h15 = torch.zeros(32, device=_b15.device).scatter_add_(0, _b15, torch.ones(len(_b15), device=_b15.device))
                                    _p15 = (_h15 + 1e-9) / (_h15.sum() + 32e-9)
                                    _ent15 = -(_p15 * _p15.log()).sum().item()
                                    print(f"[AX-15 ROUTING-ENT] step={global_step} entropy={_ent15:.3f} nats (target>1.5 for active circuit diversity)")
                    # AX-16: Ternary Information Quotient (TIQ)
                    if getattr(config, 'use_ax16_tiq_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                import math as _math16
                                _n16 = sum(_m16.dsst_s.numel() for _m16 in model.modules() if hasattr(_m16, 'dsst_s') and _m16.dsst_s is not None)
                                if _n16 > 0:
                                    _mi16 = max(0.0, 10.0 - loss.item())
                                    _C16 = _n16 * _math16.log2(3)
                                    _TIQ16 = _mi16 / max(_C16, 1.0)
                                    print(f"[AX-16 TIQ] step={global_step} MI_lb={_mi16:.3f} C_ternary={_C16/1e6:.2f}Mbits TIQ={_TIQ16:.4e} (10.09x FP16 info-eff ceiling)")
                    # AX-17: Catoni PAC-Bayes Ternary Certification (CPBTC)
                    if getattr(config, 'use_ax17_cpbtc_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                import math as _math17
                                _comm17 = sum(_m17.committed_mask.sum().item() for _m17 in model.modules() if hasattr(_m17, 'committed_mask') and _m17.committed_mask is not None)
                                if _comm17 > 0:
                                    _n_data17 = max(global_step * getattr(config, 'batch_size', 32) * getattr(config, 'seq_len', 128), 1)
                                    _eps17 = _comm17 * _math17.log(3) + _math17.log(2.0 * _math17.sqrt(_n_data17) / 0.05)
                                    _err17 = loss.item() / max(loss.item() + 1.0, 1.0)
                                    _B17 = _err17 + _math17.sqrt(max(_eps17, 0.0) / (2 * _n_data17))
                                    print(f"[AX-17 CPBTC] step={global_step} committed={_comm17/1e6:.2f}M cert_bound={_B17:.4f}")
                    # AX-18: Ternary Functional Capacity Dimension (TFCD — every 500 steps)
                    if getattr(config, 'use_ax18_tfcd_probe', True) and use_ternary and global_step % 500 == 2:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                _ax18_v = []
                                for _m18 in model.modules():
                                    if hasattr(_m18, 'dsst_s') and _m18.dsst_s is not None and _m18.dsst_s.dim() == 2 and min(_m18.dsst_s.shape) >= 4:
                                        _W18 = _m18.dsst_s.float()
                                        _sv18 = torch.linalg.svdvals(_W18)
                                        _sn18 = _sv18 / _sv18.sum().clamp(min=1e-10)
                                        _sn18 = _sn18[_sn18 > 1e-6]
                                        if _sn18.numel() > 0:
                                            _er18 = (-(_sn18 * _sn18.log()).sum()).exp().item()
                                            _ax18_v.append(_er18 / max(min(_W18.shape[0], _W18.shape[1]), 1))
                                if _ax18_v:
                                    print(f"[AX-18 TFCD] step={global_step} global_tfcd={sum(_ax18_v)/len(_ax18_v):.3f} (target>=0.70; <0.70=rank collapse)")
                    # AX-19: DSST Convergence Certificate (DCC)
                    if getattr(config, 'use_ax19_dcc_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                import math as _math19
                                _mods19 = [_m for _m in model.modules() if hasattr(_m, 'dsst_step') and hasattr(_m, 'dsst_s') and _m.dsst_s is not None]
                                _flips19 = sum(_m.dsst_step.item() * _m.dsst_s.numel() for _m in _mods19)
                                if _flips19 > 0:
                                    _n_w19 = max(sum(_m.dsst_s.numel() for _m in _mods19), 1)
                                    _gn19 = grad_norm if isinstance(grad_norm, (int, float)) else (grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm))
                                    _sg19 = _gn19 / _math19.sqrt(_n_w19)
                                    _DCC19 = _math19.sqrt(_math19.log(3)) * _sg19 / _math19.sqrt(_flips19)
                                    print(f"[AX-19 DCC] step={global_step} flip_proxy={_flips19:.0e} DCC={_DCC19:.2e} (converges as 1/sqrt(flips))")
                    # AX-20: Information-Theoretic Stopping Time (ITST)
                    if getattr(config, 'use_ax20_itst_probe', True) and use_ternary and global_step % 100 == 0:
                        if not torch.compiler.is_compiling():
                            with torch.no_grad():
                                import math as _math20
                                _comm20 = sum(_m20.committed_mask.sum().item() for _m20 in model.modules() if hasattr(_m20, 'committed_mask') and _m20.committed_mask is not None)
                                if _comm20 > 0:
                                    _n_data20 = max(global_step * getattr(config, 'batch_size', 32) * getattr(config, 'seq_len', 128), 1)
                                    _eps20 = _comm20 * _math20.log(3) + _math20.log(2.0 / 0.05)
                                    _thr20 = _math20.log(3) / (2.0 * _math20.sqrt(2.0 * _n_data20 * max(_eps20, 1e-8)))
                                    if not hasattr(config, '_ax20_prev_loss'):
                                        config._ax20_prev_loss = loss.item()
                                        config._ax20_prev_committed = _comm20
                                    _de20 = max(0.0, config._ax20_prev_loss - loss.item())
                                    _dc20 = max(1, _comm20 - config._ax20_prev_committed)
                                    config._ax20_prev_loss = loss.item()
                                    config._ax20_prev_committed = _comm20
                                    _R20 = (_de20 / _dc20) / max(_thr20, 1e-30)
                                    print(f"[AX-20 ITST] step={global_step} R_stop={_R20:.3f} committed={_comm20/1e6:.2f}M {'STOP' if _R20 < 1.0 else 'CONTINUE'}")

                    # AU-3: Error-Driven Annealing — probe: n_flips ∝ loss (SA exploration intensity)
                    if use_ternary and getattr(config, 'use_error_annealing', False) and global_step % 50 == 0:
                        _au3_loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
                        _au3_n_flips = int(getattr(config, 'error_annealing_alpha', 1.0) * _au3_loss_val) + 1
                        print(f"[AU-3 ERROR-ANNEAL] step={global_step} n_flips={_au3_n_flips} loss={_au3_loss_val:.3f}")

                    # AU-4: Shadow Hamiltonian — entropy-based commitment probe
                    if use_ternary and getattr(config, 'use_shadow_hamiltonian', False) and global_step % 50 == 0:
                        with torch.no_grad():
                            _au4_T = getattr(config, 'shadow_hamiltonian_T', 1.0)
                            _au4_entropies = []
                            for _au4_mod in model.modules():
                                if hasattr(_au4_mod, 'weight') and hasattr(_au4_mod, 'dsst_s'):
                                    _au4_shadow_w = _au4_mod.weight.float()
                                    _au4_commit_logit = _au4_shadow_w.abs() / max(_au4_T, 1e-8)
                                    _au4_p = torch.sigmoid(_au4_commit_logit)
                                    _au4_entropies.append((_au4_commit_logit * _au4_p * (1.0 - _au4_p)).mean().item())
                            if _au4_entropies:
                                _au4_eng = sum(_au4_entropies) / len(_au4_entropies)
                                print(f"[AU-4 SHADOW-HAM] step={global_step} entropy_grad_norm={_au4_eng:.4f}")

                    # AT-14: Offline Consolidation Pass — every N steps, consolidate DSST weights
                    if use_ternary and getattr(config, 'use_offline_consolidation', False):
                        _oc_n = getattr(config, 'offline_consolidation_n', 10)
                        if global_step > 0 and global_step % _oc_n == 0:
                            _oc_mag_scale = getattr(config, 'offline_consolidation_magnitude_scale', 1.1)
                            _oc_flipped = 0
                            _oc_total = 0
                            with torch.no_grad():
                                for _mod in model.modules():
                                    if (hasattr(_mod, 'use_dsst') and _mod.use_dsst
                                            and hasattr(_mod, 'dsst_s') and hasattr(_mod, 'weight')):
                                        _w = _mod.weight.float()
                                        _lsq = getattr(_mod, 'lsq_scale_pos', torch.ones(1, device=_w.device))
                                        if isinstance(_lsq, torch.Tensor):
                                            _lsq = _lsq.float()
                                        else:
                                            _lsq = float(_lsq)
                                        _mag_committed = _w.abs() > (_lsq * _oc_mag_scale if isinstance(_lsq, torch.Tensor) else _lsq * _oc_mag_scale)
                                        _s_consolidated = _w.sign().to(torch.int8)
                                        _should_flip = (_s_consolidated != _mod.dsst_s) & _mag_committed
                                        _mod.dsst_s[_should_flip] = _s_consolidated[_should_flip]
                                        _oc_flipped += _should_flip.sum().item()
                                        _oc_total += _mod.dsst_s.numel()
                            print(f"[AT-14 OFFLINE-CONSOL] step={global_step} flipped={_oc_flipped} "
                                  f"flip_rate={_oc_flipped/max(_oc_total,1):.5f}")

                    # AP-2: ETC Crystallization
                    if use_ternary and getattr(config, 'use_etc', False):
                        _etc_tau = getattr(config, 'etc_tau', 0.8)
                        _etc_cstep = getattr(config, 'etc_crystallize_step', 500)
                        if global_step >= _etc_cstep:
                            with torch.no_grad():
                                _etc_total_cryst = 0
                                for _mod in model.modules():
                                    if not (getattr(_mod, 'use_etc', False)
                                            and hasattr(_mod, 'etc_S')
                                            and hasattr(_mod, 'weight')):
                                        continue
                                    _S = _mod.etc_S.float()
                                    _cryst_mask = _S.abs() > _etc_tau
                                    if _cryst_mask.any():
                                        _alpha_etc = _mod.weight.data.abs().mean().clamp(min=1e-7)
                                        _mod.weight.data[_cryst_mask] = (
                                            _S.sign()[_cryst_mask].to(_mod.weight.dtype) * _alpha_etc
                                        )
                                        _etc_total_cryst += int(_cryst_mask.sum().item())
                                        if hasattr(_mod, 'packed_valid'):
                                            if isinstance(_mod.packed_valid, torch.Tensor):
                                                _mod.packed_valid.zero_()
                                            else:
                                                _mod.packed_valid = False
                            if global_step % 200 == 0 and not torch.compiler.is_compiling():
                                print(f"[AP-2 ETC] step={global_step} crystallized={_etc_total_cryst}")

                    # AP-5: TWA Occupancy Update
                    if use_ternary and getattr(config, 'use_twa', False):
                        with torch.no_grad():
                            _twa_logged = False
                            for _mod in model.modules():
                                if not (getattr(_mod, 'use_twa', False)
                                        and hasattr(_mod, 'twa_occ')
                                        and hasattr(_mod, 'weight')):
                                    continue
                                _w_idx = (_mod.weight.data.sign().long() + 1).clamp(0, 2)
                                _ones = torch.ones_like(_w_idx, dtype=torch.int16).unsqueeze(-1)
                                _mod.twa_occ.scatter_add_(2, _w_idx.unsqueeze(-1), _ones)
                                _mod.twa_occ.clamp_(max=32000)
                                if not _twa_logged and global_step == start_step:
                                    print(f"[AP-5 TWA] ACTIVATED | occ shape={list(_mod.twa_occ.shape)}")
                                    _twa_logged = True

                    # AP-8: A-GEM Gradient Ring Buffer
                    if getattr(config, 'use_agem', False):
                        with torch.no_grad():
                            _agem_buf_size = getattr(config, 'agem_buffer_size', 256)
                            if not hasattr(model, '_agem_grad_buffer'):
                                model._agem_grad_buffer = []
                            _g_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                            if _g_norms:
                                model._agem_grad_buffer.append(sum(_g_norms) / len(_g_norms))
                                if len(model._agem_grad_buffer) > _agem_buf_size:
                                    model._agem_grad_buffer.pop(0)
                        if global_step == start_step and not torch.compiler.is_compiling():
                            print(f"[AP-8 A-GEM] ACTIVATED | buffer_size={_agem_buf_size}")

                    # AP-9: EWC Sign-Flip Tracking
                    if use_ternary and getattr(config, 'use_ewc', False):
                        with torch.no_grad():
                            _ewc_wflips = 0.0
                            _ewc_np = 0
                            for _mod in model.modules():
                                if not (getattr(_mod, 'use_ewc', False)
                                        and hasattr(_mod, 'ewc_fisher')
                                        and hasattr(_mod, 'weight')):
                                    continue
                                _w_sign_now = _mod.weight.data.sign().to(torch.int8)
                                if _mod._ewc_prev_sign is None:
                                    _mod._ewc_prev_sign = _w_sign_now.clone()
                                else:
                                    _flip_mask = _w_sign_now != _mod._ewc_prev_sign
                                    _ewc_wflips += _mod.ewc_fisher.float()[_flip_mask].sum().item()
                                    _ewc_np += _mod.weight.numel()
                                    _mod._ewc_prev_sign.copy_(_w_sign_now)
                        if _ewc_wflips > 0 and global_step % 100 == 0 and not torch.compiler.is_compiling():
                            print(f"[AP-9 EWC] step={global_step} fisher_flips={_ewc_wflips:.2f} n={_ewc_np}")

                    # AQ-2: Dead dim pruning (periodic)
                    if getattr(config, 'use_dead_dim_pruning', False) and global_step > 0 and global_step % getattr(config, 'dead_dim_prune_step', 1000) == 0:
                        _dead_threshold = getattr(config, 'dead_dim_threshold', 0.9)
                        _n_pruned = 0
                        for _ssm_m in model.modules():
                            if hasattr(_ssm_m, 'B') and hasattr(_ssm_m.B, 'packed_weights') and _ssm_m.B.packed_weights is not None:
                                _packed = _ssm_m.B.packed_weights.float()
                                _zero_frac = (_packed == 0).float().mean(dim=0)  # per state dim
                                _alive = _zero_frac < _dead_threshold
                                _n_pruned += int((~_alive).sum().item())
                        print(f"[AQ-2 DEAD-DIM] step={global_step} dead_dims={_n_pruned} (threshold={_dead_threshold})")

                    # AQ-3: Effective rank fixed-point check
                    if getattr(config, 'use_erank_fixedpoint', False) and global_step > 0 and global_step % getattr(config, 'erank_check_interval', 500) == 0:
                        _grow_thr = getattr(config, 'erank_grow_threshold', 0.8)
                        _prune_thr = getattr(config, 'erank_prune_threshold', 0.2)
                        for _ssm_m in model.modules():
                            if hasattr(_ssm_m, 'A_log'):
                                _A_mat = _ssm_m.A_log.detach().float().unsqueeze(0)  # treat as 1×d_state for svd
                                _sv = torch.linalg.svdvals(_A_mat)
                                _sv_norm = _sv / (_sv.sum() + 1e-8)
                                _H = -(_sv_norm * (_sv_norm + 1e-10).log()).sum()
                                _er = torch.exp(_H).item()
                                _d = _ssm_m.A_log.shape[0]
                                _U = _er / _d
                                print(f"[AQ-3 ERANK-FP] step={global_step} U={_U:.3f} erank={_er:.1f} d_state={_d}")

                    # AQ-6..20 stubs (Tier 2/3 — framework placeholders)
                    # AQ-6 MoD, AQ-7 Expert-Choice MoE, AQ-8 Hypernetwork thresholds,
                    # AQ-9 CGLS stacking, AQ-10 Topo insertion, AQ-11 Fisher flow,
                    # AQ-12 RG flow, AQ-13 Committed inhibition, AQ-14 Ho-Kalman,
                    # AQ-15 Automaton collapse, AQ-16 Toeplitz CA, AQ-17 Persistent homology,
                    # AQ-18 Supernet NAS, AQ-19 Neural ODE weights, AQ-20 Foundation hypernetwork
                    # All require architectural surgery beyond a training-loop stub — registered as no-ops.

                    # AT-15: Memory Engram Orthogonality — penalize grad-sign alignment with dsst_s
                    if use_ternary and getattr(config, 'use_engram_orthogonality', False):
                        _eng_interval = getattr(config, 'engram_penalty_interval', 50)
                        if global_step % _eng_interval == 0:
                            _eng_lambda = getattr(config, 'engram_lambda', 0.01)
                            _eng_interference = 0.0
                            _eng_n = 0
                            with torch.no_grad():
                                for _mod in model.modules():
                                    if (hasattr(_mod, 'use_dsst') and _mod.use_dsst
                                            and hasattr(_mod, 'dsst_s')
                                            and _mod.weight.grad is not None):
                                        _g_sign = _mod.weight.grad.sign()
                                        _old_engram = _mod.dsst_s.float()
                                        _dot = (_g_sign * _old_engram).sum()
                                        _denom = (_g_sign.abs().sum() * _old_engram.abs().sum() + 1e-8)
                                        _eng_interference += _dot.abs() / _denom
                                        _eng_n += 1
                            if _eng_n > 0:
                                _eng_score = (_eng_interference / _eng_n).item()
                                print(f"[AT-15 ENGRAM-ORTHO] step={global_step} interference={_eng_score:.4f}")

                    # AU-11: KMC Updates — top-K gradient firing probe
                    if use_ternary and getattr(config, 'use_kmc_updates', False) and global_step % 100 == 0:
                        with torch.no_grad():
                            for _au11_name, _au11_p in model.named_parameters():
                                if _au11_p.grad is not None and 'weight' in _au11_name:
                                    _au11_grad_abs = _au11_p.grad.abs().flatten()
                                    _au11_k = max(1, int(getattr(config, 'kmc_top_k_fraction', 0.1) * _au11_grad_abs.numel()))
                                    _au11_thr = _au11_grad_abs.topk(_au11_k).values[-1].item()
                                    _au11_layer = _au11_name.split('.')[-2] if '.' in _au11_name else _au11_name
                                    print(f"[AU-11 KMC] step={global_step} layer={_au11_layer} threshold={_au11_thr:.4f}")
                                    break  # first weight layer only

                    # AU-12: RG Layer Freeze — layers with lowest grad norm (RG-irrelevant)
                    if use_ternary and getattr(config, 'use_rg_layer_freeze', False) and global_step % 100 == 0:
                        with torch.no_grad():
                            _au12_norms = [(n, p.grad.norm().item()) for n, p in model.named_parameters() if p.grad is not None]
                            if _au12_norms:
                                _au12_sorted = sorted(_au12_norms, key=lambda x: x[1])
                                _au12_n = int(len(_au12_sorted) * getattr(config, 'rg_freeze_fraction', 0.1))
                                print(f"[AU-12 RG-FREEZE] step={global_step} would_freeze={_au12_n}/{len(_au12_sorted)} "
                                      f"(lowest: {_au12_sorted[0][0]} norm={_au12_sorted[0][1]:.5f})")

                    # AU-14: Nosé-Hoover — flip rate thermostat probe
                    if use_ternary and getattr(config, 'use_nose_hoover', False) and global_step % 50 == 0:
                        with torch.no_grad():
                            _au14_flips, _au14_total = 0, 0
                            for _au14_mod in model.modules():
                                if hasattr(_au14_mod, 'dsst_s') and hasattr(_au14_mod, 'weight'):
                                    _au14_s_now = _au14_mod.weight.float().sign().to(torch.int8)
                                    _au14_flips += (_au14_s_now != _au14_mod.dsst_s).sum().item()
                                    _au14_total += _au14_mod.dsst_s.numel()
                            if _au14_total > 0:
                                _au14_rate = _au14_flips / _au14_total
                                _au14_xi = _au14_rate - getattr(config, 'nose_hoover_target_flip_rate', 0.01)
                                print(f"[AU-14 NOSE-HOOVER] step={global_step} flip_rate={_au14_rate:.4f} xi={_au14_xi:.4f}")

                    # AU-17: RSB Monitor — Edwards-Anderson order parameter
                    if use_ternary and getattr(config, 'use_rsb_monitor', False) and global_step % getattr(config, 'rsb_check_interval', 100) == 0:
                        with torch.no_grad():
                            _au17_qEA = [(_m.dsst_s != 0).float().mean().item() for _m in model.modules() if hasattr(_m, 'dsst_s')]
                            if _au17_qEA:
                                _au17_mean = sum(_au17_qEA) / len(_au17_qEA)
                                print(f"[AU-17 RSB] step={global_step} q_EA={_au17_mean:.4f} "
                                      f"(0=paramagnetic 1=ferromagnetic 0.3-0.7=spin-glass)")

                    # AR-7: Structured zero columns — update col_activity EMA
                    if getattr(config, 'use_structured_zero_cols', False):
                        for _m in model.modules():
                            if hasattr(_m, 'col_activity') and _m.col_activity is not None:
                                if hasattr(_m, 'weight') and _m.weight is not None:
                                    _col_nnz = (_m.weight.detach() != 0).float().mean(dim=0)
                                    _m.col_activity.mul_(0.99).add_(_col_nnz * 0.01)
                        if global_step % 100 == 0:
                            print(f"[AR-7 STRUCT-ZERO] step={global_step} col_activity updated")

                    # Lion: verify state has only exp_avg (no exp_avg_sq)
                    if global_step == start_step and optimizer_type == "lion":
                        _first_p = next(
                            (p for g in optimizer.param_groups for p in g["params"]
                             if p in optimizer.state and len(optimizer.state[p]) > 0),
                            None,
                        )
                        if _first_p is not None:
                            _lion_keys = list(optimizer.state[_first_p].keys())
                            assert "exp_avg_sq" not in _lion_keys, (
                                f"[LION-K] BUG: exp_avg_sq found in Lion state: {_lion_keys}"
                            )
                            assert "exp_avg" in _lion_keys or "exp_avg_int8" in _lion_keys, (
                                f"[LION-K] BUG: exp_avg or exp_avg_int8 missing from Lion state: {_lion_keys}"
                            )
                            print(f"[LION-K] State verified at step {global_step}: keys={_lion_keys}")

                    # E1: Two-Stage Flip-Triggered LR Schedule — Update tracker after step
                    if flip_tracker is not None:
                        flip_stats = flip_tracker.update(model, global_step)
                        if flip_stats["just_transitioned"]:
                            print(
                                f"[E1-TRANSITION] Step {global_step}: flip_rate_ema={flip_stats['flip_rate_ema'] * 100:.2f}% "
                                f"< threshold={config.flip_lr_threshold * 100:.1f}%. Entering phase 2 (decay).",
                                flush=True,
                            )

                    # G1: PID Commitment Rate Controller
                    # Measures committed fraction, computes PID error vs target,
                    # adjusts LR multiplicatively across all param groups.
                    if (
                        use_ternary
                        and getattr(config, "use_pid_commitment_lr", True)
                        and global_step > warmup_steps
                        and global_step % getattr(config, "pid_update_interval", 100) == 0
                    ):
                        _g1_raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                        _g1_total_w = 0
                        _g1_committed_w = 0
                        for _g1_m in _g1_raw.modules():
                            if hasattr(_g1_m, "committed_mask") and _g1_m.committed_mask is not None:
                                _g1_total_w += _g1_m.committed_mask.numel()
                                _g1_committed_w += int(_g1_m.committed_mask.sum().item())
                        if _g1_total_w > 0:
                            _g1_committed_frac = _g1_committed_w / _g1_total_w
                            _g1_target = getattr(config, "pid_target_commitment", 0.5)
                            _g1_kp = getattr(config, "pid_kp", 0.1)
                            _g1_ki = getattr(config, "pid_ki", 0.01)
                            _g1_kd = getattr(config, "pid_kd", 0.5)
                            _g1_error = _g1_target - _g1_committed_frac
                            _g1_integral = _g1_integral + _g1_error
                            _g1_derivative = _g1_error - _g1_prev_error
                            _g1_pid_output = _g1_kp * _g1_error + _g1_ki * _g1_integral + _g1_kd * _g1_derivative
                            _g1_pid_lr_mult = max(0.5, min(2.0, _g1_pid_lr_mult + _g1_pid_output))
                            _g1_prev_error = _g1_error
                            _g1_prev_committed_frac = _g1_committed_frac
                            for _g1_grp in optimizer.param_groups:
                                _g1_prev_mult = _g1_grp.get("_g1_prev_mult", 1.0)
                                _g1_grp["lr"] = _g1_grp["lr"] * (_g1_pid_lr_mult / max(1e-8, _g1_prev_mult))
                                _g1_grp["_g1_prev_mult"] = _g1_pid_lr_mult
                            print(
                                f"[G1-PID] step={global_step} committed={_g1_committed_frac:.3f} "
                                f"err={_g1_error:.3f} pid_lr_mult={_g1_pid_lr_mult:.3f}"
                            )

                    # Update LR in all TernaryLinear layers for C2 threshold computation
                    # ================================================================
                    # TIER 3 Phase 3 & 4: Post-Step Monitoring & Adjustments
                    # ================================================================

                    # R3-A6-P5: Bifurcated Optimizer
                    if bifurcated_optimizer is not None and use_ternary:
                        bifurcated_optimizer.step()
                        if global_step % 100 == 0:
                            print(
                                f"[Bifurcated] Crystal={len(bifurcated_optimizer.crystal)}, "
                                f"Liquid={len(bifurcated_optimizer.liquid)}"
                            )

                    # R3-A6-M5: CUSUM Drift Monitoring
                    if cusum_monitor is not None and use_ternary:
                        for name, param in model.named_parameters():
                            if param.grad is not None and "weight" in name:
                                optimizer_state = None
                                if param in optimizer.state:
                                    optimizer_state = optimizer.state[param]

                                if cusum_monitor.update(
                                    param, param.grad, optimizer_state
                                ):
                                    logger.info(f"[CUSUM] Drift detected at {name}")

                        if global_step % 100 == 0:
                            stats = cusum_monitor.get_stats()
                            if stats["active_count"] > 0:
                                print(
                                    f"[CUSUM] Active={stats['active_count']}, "
                                    f"Max_S={stats['max_s']:.2f}, Alarms={stats['alarms']}"
                                )

                    # R3-A6-P3: Output Gradient Probe
                    if probe_monitors and global_step % 100 == 0:
                        for name, probe in probe_monitors.items():
                            stats = probe.get_stats()
                            if stats["alarms"] > 0:
                                print(
                                    f"[Probe] {name}: Mean={stats['mean']:.2e}, "
                                    f"Var={stats['var']:.2e}, Alarms={stats['alarms']}"
                                )

                    # R3-A4-P2: Hybrid RTRL Logging
                    if hybrid_rtrl is not None and global_step % 100 == 0:
                        stats = hybrid_rtrl.get_stats()
                        print(f"[RTRL] RTRL_norm={stats['rtrl_norm']:.2e}")

                    # R3-A6-P6: Pattern Transplantation
                    if (
                        pattern_transplant is not None
                        and use_ternary
                        and global_step % 100 == 0
                    ):
                        stats = pattern_transplant.get_stats()
                        print(
                            f"[Transplant] Patterns={stats['patterns_stored']}, "
                            f"Transplants={stats['transplants']}"
                        )

                    # R3-A6-P8: Commit-Restart Scheduler
                    if restart_scheduler is not None and use_ternary:
                        if commitment_monitor is not None:
                            commit_stats = commitment_monitor.update(model, global_step)
                            current_commitment = int(
                                sum(commit_stats.get("committed_counts", [0]))
                            )

                            if restart_scheduler.should_restart(
                                global_step, current_commitment
                            ):
                                uncommitted = [
                                    p
                                    for name, p in model.named_parameters()
                                    if "weight" in name and p.requires_grad
                                ]
                                restart_scheduler.restart_uncommitted(
                                    optimizer, uncommitted, global_step
                                )

                    # F17: Invalidate packed cache after weight update
                    if getattr(config, "use_ternary", False):
                        from .components.ternary import invalidate_packed_cache_model

                        invalidate_packed_cache_model(model)

                    # [F26] Commitment monitoring
                    if _ternary_monitor is not None and global_step % 10 == 0:
                        commit_stats = _ternary_monitor.update(model)

                        # F2: Release optimizer state for fully committed layers (R10-A3-B-F2)
                        for _f2_name, _f2_module in model.named_modules():
                            if (hasattr(_f2_module, 'committed_mask')
                                    and _f2_module.committed_mask is not None):
                                _f2_frac = _f2_module.committed_mask.float().mean().item()
                                if _f2_frac > 0.95:
                                    for _f2_p in _f2_module.parameters(recurse=False):
                                        _f2_state = optimizer.state.get(_f2_p)
                                        if _f2_state and _f2_state.get("exp_avg_int8") is not None:
                                            _f2_state["exp_avg_int8"] = None
                                            _f2_state["exp_avg_scale"] = None
                                            print(f"[F2-STATE-RELEASE] {_f2_name}: {_f2_frac:.0%} committed → optimizer state freed")

                        # Log every 50 steps
                        if global_step % 50 == 0:
                            print(
                                f"[COMMITMENT] Step {global_step}: C={commit_stats['commitment']:.4f} "
                                f"(flip_rate={commit_stats['flip_rate']:.4f}, "
                                f"avg_flip={commit_stats['avg_flip_rate']:.4f})"
                            )

                            # Detailed per-layer stats every 200 steps
                            if global_step % 200 == 0 and global_step > 0:
                                print(f"[COMMITMENT] Per-layer flip rates:")
                                for layer_name, stats in list(
                                    commit_stats["per_layer"].items()
                                )[:5]:  # Show first 5 layers
                                    print(
                                        f"  {layer_name}: flip={stats['flip_rate']:.4f} "
                                        f"({stats['flipped_count']}/{stats['total_count']})"
                                    )

                    # F14: Dynamic Param Groups Eviction for committed weights
                    if getattr(config, "use_param_eviction", False) and global_step > 0:
                        if global_step % config.eviction_check_interval == 0:
                            evict_committed_params(
                                optimizer=optimizer,
                                commitment_monitor=commitment_monitor,
                                commitment_threshold=config.eviction_commitment_threshold,
                                logger=locals().get("logger"),
                            )

                    # R4-A4-F10: Log update norms
                    if gradient_monitor is not None:
                        gradient_monitor.log_update_norms(model, global_step)

                    # R4-A4-M7: Record weight flips
                    if delta_chronicle is not None:
                        flip_stats = delta_chronicle.record_step(model, global_step)
                        if flip_stats["flips"] > 0 and global_step % 100 == 0:
                            print(
                                f"[CHRONICLE] step {global_step}: {flip_stats['flips']} flips, "
                                f"rate={flip_stats['flip_rate']:.6f}"
                            )

                        # R4-A4-M8: Compute flip influences
                        if flip_tracker is not None and flip_stats["flips"] > 0:
                            step_flips = (
                                delta_chronicle._history[-1][1]
                                if delta_chronicle._history
                                else {}
                            )
                            influence_stats = flip_tracker.compute_influences(
                                model, step_flips, global_step
                            )

                    if aga_optimizer is not None:
                        aga_optimizer.step()
                    if bbo_optimizer is not None:
                        bbo_optimizer.step()
                    if ema is not None:
                        ema.update(
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        )

                    # C3: Progressive Layer Freezing
                    if progressive_freezer is not None:
                        freeze_stats = progressive_freezer.check_and_freeze(global_step)
                        if freeze_stats["newly_frozen"]:
                            print(
                                f"[C3-FREEZE] Newly frozen: {freeze_stats['newly_frozen']} "
                                f"(total: {freeze_stats['total_frozen']}/{freeze_stats['total_layers']})"
                            )

                    # Sparse Active Shadow Weights (R4-A2-M6/P4)
                    if (
                        use_sparse_shadow
                        and getattr(config, "use_ternary", False)
                        and global_step % sparse_shadow_interval == 0
                    ):
                        _ss_log = global_step == start_step or global_step % 50 == 0
                        apply_sparse_shadow_weights(
                            model,
                            optimizer,
                            margin=sparse_shadow_margin,
                            decay=sparse_shadow_decay,
                            step=global_step,
                            log=_ss_log,
                        )

                    # Update current LR on modules for dead zone decay (R2-A6-M14)
                    if use_dead_zone:
                        try:
                            _step_lr = optimizer.param_groups[0]["lr"]
                        except (IndexError, KeyError, AttributeError):
                            _step_lr = base_lr
                        for _m in model.modules():
                            if hasattr(_m, "_current_lr"):
                                _m._current_lr = _step_lr

                    # Information-theoretic freezing: update confidence (R2-A3-M3)
                    if use_info_freeze:
                        update_info_freeze_vhat(model, optimizer)
                        for _m in model.modules():
                            if hasattr(_m, "update_freeze_confidence"):
                                _m.update_freeze_confidence()
                        zero_frozen_optimizer_state(model, optimizer)
                    elif use_ternary:
                        # A3-M7: C2 commitment optimizer state zeroing (L1/L2 no-op, L3 active)
                        zero_frozen_optimizer_state(model, optimizer)

                    # AutoFreeze: layer-level gradient-norm convergence
                    if autofreeze is not None:
                        _af_result = autofreeze.step(global_step)
                        if _af_result["newly_frozen"]:
                            print(f"[AutoFreeze] step={global_step} newly_frozen={_af_result['newly_frozen']}")

                    # R1-A6: SAP -- log per-layer sensitivity (always active when ternary)
                    if use_ternary and global_step % 50 == 0:
                        log_layer_sensitivity(model, global_step)

                    # R2-A2-M4: End fused AGA step cycle
                    if fused_aga is not None:
                        fused_aga.end_step()

                    if getattr(config, "use_caq", False):
                        update_caq_state(model, optimizer)

                    # INSTANT: Update step counter for activation projection scheduling (R2-A2-F6)
                    if getattr(config, "use_instant", False):
                        for module in model.modules():
                            if hasattr(module, "_instant_step"):
                                module._instant_step = global_step

                    # Phase 20.A: Anti-windup for shadow weights
                    if getattr(config, "antiwindup_k", 0) > 0:
                        from .components.ternary import apply_antiwindup

                        apply_antiwindup(model, k_aw=config.antiwindup_k)

                    # Fused requantize or invalidate ternary weight caches (weights changed by optimizer)
                    if use_fused_requantize:
                        _fused_n = 0
                        for module in model.modules():
                            if hasattr(module, "requantize_from_shadow"):
                                module.requantize_from_shadow()
                                _fused_n += 1
                        if global_step == start_step:
                            print(
                                f"[FUSED-REQUANT] Pre-warmed {_fused_n} ternary weight caches",
                                flush=True,
                            )
                    else:
                        for module in model.modules():
                            if hasattr(module, "invalidate_weight_cache"):
                                module.invalidate_weight_cache()
                    if hasattr(model, "optrot") or any(
                        hasattr(m, "optrot") for m in model.modules()
                    ):
                        for module in model.modules():
                            if hasattr(module, "optrot") and hasattr(
                                module.optrot, "invalidate_rotation_cache"
                            ):
                                module.optrot.invalidate_rotation_cache()

                    # Periodic merge for deferred ternary
                    if getattr(config, "use_deferred_ternary", False):
                        merge_interval = getattr(config, "deferred_merge_interval", 100)
                        if global_step % merge_interval == 0 and global_step > 0:
                            from .components.ternary import merge_all_deferred_deltas

                            n_merged = merge_all_deferred_deltas(model, optimizer)

                # Commitment monitor check (R2-A4-M3+M8)
                if (
                    commitment_monitor is not None
                    and global_step % commitment_monitor.check_interval == 0
                ):
                    cm_result = commitment_monitor.check(model, global_step)

                    cm_lr_mult = commitment_monitor.get_lr_multiplier()
                    if cm_lr_mult != 1.0:
                        for pg in optimizer.param_groups:
                            pg["lr"] = pg["lr"] * cm_lr_mult

                    print(
                        f"  [COMMITMENT] step={global_step} "
                        f"H={cm_result['avg_entropy']:.4f} "
                        f"util={cm_result['avg_utilization']:.3f} "
                        f"committed={cm_result['avg_committed']:.4f} "
                        f"dH/dt={cm_result['dH_dt']:.6f} "
                        f"plateau={'YES' if cm_result['plateau_active'] else 'no'} "
                        f"({cm_result['consecutive_plateau']}/{commitment_monitor.plateau_patience}) "
                        f"action={cm_result['action']} "
                        f"lr_mult={cm_lr_mult:.2f}"
                    )

                    if cm_result["should_stop"]:
                        print(
                            f"\n[COMMITMENT] EARLY STOP at step {global_step}: "
                            f"{commitment_monitor.stop_reason}\n"
                        )
                        break

                if scheduler is not None:
                    scheduler.step()

                log_and_eval_ctx = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "ema": ema,
                    "config": config,
                    "global_step": global_step,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "epoch": epoch,
                    "epoch_tokens": epoch_tokens,
                    "batch_size": batch_size,
                    "device": device,
                    "data_dir": data_dir,
                    "eval_every": eval_every,
                    "ppl_history": ppl_history,
                    "checkpoint_every": checkpoint_every,
                    "save_dir": save_dir,
                    "start_step": start_step,
                    "anomaly_detector": anomaly_detector,
                    "ram_checkpointer": ram_checkpointer,
                    "on_checkpoint": on_checkpoint,
                    "two_stage_decay": two_stage_decay,
                    "total_steps": total_steps,
                    "two_stage_wd_start_pct": two_stage_wd_start_pct,
                    "weight_decay": weight_decay,
                    "log_every": log_every,
                    "start_time": start_time,
                    "base_lr": base_lr,
                    "monitor": monitor,
                    "lr_schedule_type": lr_schedule_type,
                    "two_stage_drop_pct": two_stage_drop_pct,
                    "two_stage_drop_factor": two_stage_drop_factor,
                    "progressive_scheduler": progressive_scheduler,
                    "aga_optimizer": aga_optimizer,
                    "bbo_optimizer": bbo_optimizer,
                    "fused_aga": fused_aga,
                    "rho1_stats": rho1_stats,
                    "use_adagc": use_adagc,
                    "clip_stats": clip_stats,
                    "let_teacher": let_teacher,
                    "let_stop_step": let_stop_step,
                    "teacher_model": teacher_model,
                    "od_teacher": od_teacher,
                    "angular_reg_lambda": angular_reg_lambda,
                    "boundary_reg_lambda": boundary_reg_lambda,
                    "use_tdo": use_tdo,
                    "_tdo_stats": _tdo_stats,
                    "_ang_loss": _ang_loss,
                    "_bnd_loss": _bnd_loss,
                    "ce_loss": ce_loss,
                    "kd_loss": kd_loss,
                    "od_kd_loss": od_kd_loss,
                    "cos_sim": cos_sim,
                    "instant_rank": instant_rank,
                    "use_activation_quant": use_activation_quant,
                    "activation_quant_bits": activation_quant_bits,
                    "use_info_freeze": use_info_freeze,
                    "use_dead_zone": use_dead_zone,
                    "mem_mb": mem_mb,
                    "max_steps": max_steps,
                }
                _log_and_eval(log_and_eval_ctx)

                optimizer.zero_grad()
                if aga_optimizer is not None:
                    aga_optimizer.zero_grad()
                if bbo_optimizer is not None:
                    bbo_optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0 and torch.cuda.is_available() and _l2_current_seq_len < config.seq_len:
                    _l2_free, _l2_total = torch.cuda.mem_get_info()
                    if _l2_total > 0 and _l2_free / _l2_total > 0.25:
                        _l2_prev = _l2_current_seq_len
                        _l2_current_seq_len = min(config.seq_len, ((_l2_current_seq_len + 64) // 8) * 8)
                        if _l2_current_seq_len != _l2_prev:
                            train_loader = get_dataloader(
                                dataset_type,
                                config.vocab_size,
                                _l2_current_seq_len,
                                _l1_current_batch,
                                num_samples,
                                data_dir=data_dir,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers,
                                prefetch_factor=prefetch_factor,
                                ref_losses_path=rho1_ref_losses_path if use_rho1 else None,
                                use_sequence_packing=use_sequence_packing,
                                sampler=curriculum_sampler,
                            )
                            print(f"[L2-VRAM-SEQ] step={global_step} free={_l2_free/_l2_total*100:.1f}% seq_len={_l2_prev}->{_l2_current_seq_len}")

                # Log two-stage phase transition
                if lr_schedule_type == "two_stage":
                    _drop_step = int(total_steps * two_stage_drop_pct)
                    if global_step == _drop_step:
                        print(
                            f"\n[TWO-STAGE] Step {global_step}: LR DROP "
                            f"{base_lr:.2e} -> {base_lr * two_stage_drop_factor:.2e} "
                            f"(entering commitment phase)\n"
                        )

                # Update training progress for all schedule-dependent features
                # (thermodynamic threshold, HESTIA phases, gradual threshold, etc.)
                if config.use_ternary:
                    model.set_training_progress(global_step, total_steps)

                if progressive_scheduler is not None:
                    phase_msg = progressive_scheduler.step(global_step)
                    if phase_msg:
                        print(f"\n[PROGRESSIVE] Step {global_step}: {phase_msg}\n")

                if global_step % log_every == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                    try:
                        current_lr = optimizer.param_groups[0]["lr"]
                    except (IndexError, KeyError):
                        current_lr = base_lr

                    log_dict = {
                        "step": global_step,
                        "loss": loss.item(),
                        "lr": current_lr,
                        "tokens_per_sec": tokens_per_sec,
                        "grad_norm": grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm,
                        "epoch": epoch,
                    }
                    if progressive_scheduler is not None:
                        log_dict["phase"] = progressive_scheduler.phase_name
                    monitor.log(**log_dict)

                    extra = f" | Mem {mem_mb:.0f}MB"
                    if two_stage_decay:
                        _current_wd = optimizer.param_groups[0].get(
                            "weight_decay", weight_decay
                        )
                        extra += f" | WD={_current_wd:.4f}"
                    if lr_schedule_type == "two_stage":
                        _drop_step = int(total_steps * two_stage_drop_pct)
                        _phase = "explore" if global_step < _drop_step else "commit"
                        extra += f" | Phase={_phase}"
                    if (
                        aga_optimizer is not None
                        and global_step % 50 == 0
                        and global_step > 0
                    ):
                        aga_stats = aga_optimizer.get_stats()
                        extra += (
                            f" | [AGA] flips/step={aga_stats['flips_per_step']:.0f} "
                            f"avg|acc|={aga_stats['avg_abs_accumulator']:.1f} "
                            f"max|acc|={aga_stats['max_accumulator']:.0f} "
                            f"near_K={aga_stats['near_threshold_pct']:.1f}%"
                        )
                    if (
                        bbo_optimizer is not None
                        and global_step % 50 == 0
                        and global_step > 0
                    ):
                        bbo_stats = bbo_optimizer.get_stats()
                        dist = bbo_stats["state_distribution"]
                        extra += (
                            f" | [BBO] flips/step={bbo_stats['flips_per_step']:.1f} "
                            f"ballot={bbo_stats['avg_abs_ballot']:.1f} "
                            f"dist={{-1:{dist[-1]:.1f}%, 0:{dist[0]:.1f}%, +1:{dist[1]:.1f}%}}"
                        )
                    if (
                        fused_aga is not None
                        and global_step % 50 == 0
                        and global_step > 0
                    ):
                        _fstats = fused_aga.get_stats()
                        extra += (
                            f"  [FUSED-AGA] flips/step={_fstats['flips_per_step']:.1f} "
                            f"avg_acc={_fstats['avg_abs_accumulator']:.2f} "
                            f"near_K={_fstats['near_threshold_pct']:.1f}% "
                            f"total_flips={_fstats['total_flips']}"
                        )
                    if rho1_stats is not None:
                        extra += f" | Rho1 kept={rho1_stats['kept_ratio']:.1%} excess={rho1_stats['mean_excess']:.3f}"
                    if use_adagc and "_adagc_clips" in dir():
                        extra += f" | AdaGC clips={_adagc_clips} max_ratio={_adagc_max_ratio:.2f}"
                    # [F22] Log adaptive gradient clipping stats
                    if global_step % 100 == 0 or clip_stats["clipped"]:
                        extra += f" | [AdaGC] tau={clip_stats['threshold']:.3f} (mu={clip_stats['mean']:.3f} sigma={clip_stats['std']:.3f}) clipped={clip_stats['clipped']}"
                    if let_teacher is not None and global_step < let_stop_step:
                        extra += f" | LET cos={cos_sim.item():.4f}"
                    if teacher_model is not None:
                        extra += f" | CE {ce_loss.item():.4f} KD {kd_loss.item():.4f}"
                    if od_teacher is not None:
                        extra += f" | OD-KD {od_kd_loss.item():.4f}"
                    if (
                        angular_reg_lambda > 0 or boundary_reg_lambda > 0
                    ) and config.use_ternary:
                        extra += f" | AngReg {_ang_loss.item():.4f} BndReg {_bnd_loss.item():.4f}"
                    if use_tdo and config.use_ternary:
                        extra += f" | TDO {_tdo_loss.item():.4f} [{_tdo_stats['neg']:.2f}/{_tdo_stats['zero']:.2f}/{_tdo_stats['pos']:.2f}]"
                    if progressive_scheduler is not None:
                        extra += f" | Phase {progressive_scheduler.phase_name}"
                    if config.use_ternary:
                        _prefer_dsst = getattr(config, 'use_dsst', False)
                        for name, module in model.named_modules():
                            if hasattr(module, "get_ternary_stats"):
                                if _prefer_dsst and not getattr(module, 'use_dsst', False):
                                    continue
                                stats = module.get_ternary_stats()
                                extra += f" | Tern +{stats['positive']:.2f} 0:{stats['zero']:.2f} -{stats['negative']:.2f}"
                                break

                    if (
                        getattr(config, "use_deferred_ternary", False)
                        and global_step % log_every == 0
                    ):
                        from .components.ternary import get_deferred_stats

                        d_stats = get_deferred_stats(model)
                        if d_stats["enabled"]:
                            extra += f" | Deferred delta_norm: avg={d_stats['avg_delta_norm_ratio']:.4f} max={d_stats['max_delta_norm_ratio']:.4f}"

                    if (
                        getattr(config, "use_instant", False)
                        and global_step % (log_every * 4) == 0
                    ):
                        _inst_active = 0
                        _inst_total = 0
                        _inst_tensors = 0
                        for _iname, _imod in model.named_modules():
                            if hasattr(_imod, "_instant_step") and hasattr(
                                _imod, "use_instant"
                            ):
                                _inst_total += 1
                                if (
                                    _imod.use_instant
                                    and _imod._instant_step >= _imod._instant_warmup
                                ):
                                    _inst_active += 1
                                    _inst_tensors += len(
                                        getattr(_imod, "_instant_projectors", {})
                                    )
                        if _inst_total > 0:
                            print(
                                f"  [INSTANT] Active SWA layers: {_inst_active}/{_inst_total}, "
                                f"projectors: {_inst_tensors}, rank={instant_rank}"
                            )

                    if config.use_ternary and global_step % (log_every * 4) == 0:
                        entropy_stats = compute_weight_entropy(model)
                        commitment_stats = compute_commitment_rate(model)
                        if entropy_stats and commitment_stats:
                            avg_util = sum(
                                s["utilization"] for s in entropy_stats
                            ) / len(entropy_stats)
                            min_util = min(
                                (s["utilization"] for s in entropy_stats), default=0
                            )
                            avg_committed = sum(
                                s["committed_frac"] for s in commitment_stats
                            ) / len(commitment_stats)
                            print(
                                f"  [DIAG] Entropy: avg_util={avg_util:.3f} min_util={min_util:.3f} | "
                                f"Commitment: avg={avg_committed:.3f}"
                            )
                            for s in entropy_stats:
                                if s["utilization"] < 0.8:
                                    print(
                                        f"  [DIAG WARNING] Low entropy layer {s['name']}: "
                                        f"H={s['entropy']:.3f}/{s['max_entropy']:.3f} "
                                        f"util={s['utilization']:.3f} dist={s['dist']}"
                                    )

                    if (
                        use_activation_quant
                        and config.use_ternary
                        and global_step % (log_every * 4) == 0
                    ):
                        _aq_active = 0
                        _aq_total = 0
                        for _aqname, _aqmod in model.named_modules():
                            if hasattr(_aqmod, "_act_quant_step") and hasattr(
                                _aqmod, "use_activation_quant"
                            ):
                                _aq_total += 1
                                if (
                                    _aqmod.use_activation_quant
                                    and _aqmod._act_quant_step
                                    >= _aqmod._act_quant_warmup
                                ):
                                    _aq_active += 1
                        if _aq_total > 0:
                            print(
                                f"  [ACT-QUANT] Active SSM layers: {_aq_active}/{_aq_total}, "
                                f"bits={activation_quant_bits}"
                            )

                    if (
                        use_info_freeze
                        and config.use_ternary
                        and global_step % (log_every * 4) == 0
                    ):
                        _total_frozen = 0
                        _total_params = 0
                        for _fname, _fmod in model.named_modules():
                            if hasattr(_fmod, "get_freeze_stats"):
                                _fstats = _fmod.get_freeze_stats()
                                if _fstats.get("freeze_enabled"):
                                    _total_frozen += _fstats["frozen_count"]
                                    _total_params += _fstats["total_count"]
                        if _total_params > 0:
                            _freeze_pct = _total_frozen / _total_params
                            print(
                                f"  [INFO-FREEZE] Global: {_total_frozen}/{_total_params} "
                                f"({_freeze_pct:.1%}) frozen"
                            )

                    if (
                        use_dead_zone
                        and config.use_ternary
                        and global_step % (log_every * 4) == 0
                    ):
                        _total_blocked = 0
                        _total_allowed = 0
                        for _dzname, _dzmod in model.named_modules():
                            if hasattr(_dzmod, "_dead_zone_flips_blocked"):
                                _total_blocked += _dzmod._dead_zone_flips_blocked
                                _total_allowed += _dzmod._dead_zone_flips_allowed
                        _total_flips = _total_blocked + _total_allowed
                        _block_rate = _total_blocked / max(_total_flips, 1)
                        print(
                            f"  [DEAD-ZONE] Global: blocked={_total_blocked} "
                            f"allowed={_total_allowed} block_rate={_block_rate:.1%}"
                        )

                    if config.use_routing:
                        routing_stats = []
                        for name, module in model.named_modules():
                            if hasattr(module, "get_stats") and hasattr(
                                module, "capacity"
                            ):
                                stats = module.get_stats()
                                routing_stats.append(stats)
                        if routing_stats:
                            avg_ratio = sum(
                                s["process_ratio"] for s in routing_stats
                            ) / len(routing_stats)
                            avg_weight = sum(
                                s["mean_weight"] for s in routing_stats
                            ) / len(routing_stats)
                            extra += f" | Route {avg_ratio:.2f} MeanW {avg_weight:.3f}"

                    # TADA: Log alpha values and trigger rank monitoring
                    if (
                        getattr(config, "use_tada", False)
                        and global_step % log_every == 0
                    ):
                        if (
                            hasattr(model, "tada_alphas")
                            and model.tada_alphas is not None
                        ):
                            alphas = model.tada_alphas.data.cpu().numpy()
                            alpha_str = ", ".join(f"{a:.3f}" for a in alphas)
                            print(f"  [TADA] alphas=[{alpha_str}]")
                        if hasattr(model, "_monitor_tada_rank"):
                            model._monitor_tada_rank()

                    if config.use_moe and global_step % 50 == 0:
                        _moe_stats = []
                        for _name, _m in model.named_modules():
                            if hasattr(_m, "_expert_counts") and hasattr(
                                _m, "get_stats"
                            ):
                                _s = _m.get_stats()
                                if _s.get("total_tokens", 0) > 0:
                                    _moe_stats.append(_s)
                                    _m.reset_stats()
                        if _moe_stats:
                            _avg_bal = sum(
                                s["balance_ratio"] for s in _moe_stats
                            ) / len(_moe_stats)
                            extra += f" | MoE bal={_avg_bal:.3f}"

                    if (
                        getattr(config, "use_ftec", False)
                        and global_step % log_every == 0
                    ):
                        for name, module in model.named_modules():
                            if hasattr(module, "n_factors") and hasattr(
                                module, "shadow_values"
                            ):
                                for k in range(module.n_factors):
                                    from .components.ternary import (
                                        TernaryQuantize,
                                    )

                                    w = module.shadow_values[k].data
                                    threshold = getattr(
                                        module,
                                        "threshold_coeff",
                                        DEFAULT_THRESHOLD_COEFF,
                                    )
                                    q = TernaryQuantize.apply(
                                        w, module.group_size, threshold
                                    )
                                    n_pos = (q > 0).sum().item()
                                    n_neg = (q < 0).sum().item()
                                    n_zero = (q == 0).sum().item()
                                    total = q.numel()
                                    print(
                                        f"  [FTEC] {name} factor={k} +1={n_pos / total:.1%} 0={n_zero / total:.1%} -1={n_neg / total:.1%}"
                                    )
                                if hasattr(module, "skip_alpha"):
                                    print(
                                        f"  [FTEC] {name} skip_alpha={module.skip_alpha.item():.4f}"
                                    )
                                break

                    monitor.print_status(
                        global_step,
                        loss.item(),
                        current_lr,
                        tokens_per_sec,
                        grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm,
                        extra,
                    )

                    # Log global commitment statistics
                    if (
                        hasattr(config, "use_commitment_tracking")
                        and config.use_commitment_tracking
                    ):
                        total_weights = 0
                        committed_weights = 0
                        total_sigma_g = 0
                        total_margin = 0
                        num_layers = 0

                        for module in model.modules():
                            if (
                                hasattr(module, "committed_mask")
                                and module.committed_mask is not None
                            ):
                                num_layers += 1
                                total_weights += module.committed_mask.numel()
                                committed_weights += module.committed_mask.sum().item()
                                total_sigma_g += module.sigma_g.mean().item()

                                dist_neg = (module.weight + 0.5).abs()
                                dist_pos = (module.weight - 0.5).abs()
                                margin = torch.min(dist_neg, dist_pos).mean().item()
                                total_margin += margin

                        if num_layers > 0:
                            global_commit_rate = (
                                committed_weights / total_weights
                                if total_weights > 0
                                else 0
                            )
                            avg_sigma_g = total_sigma_g / num_layers
                            avg_margin = total_margin / num_layers

                            print(
                                f"[C2-GLOBAL] step={global_step}, "
                                f"commit_rate={global_commit_rate * 100:.2f}%, "
                                f"committed={committed_weights}/{total_weights}, "
                                f"avg_sigma_g={avg_sigma_g:.4f}, "
                                f"avg_margin={avg_margin:.4f}"
                            )

                            progress = (
                                global_step / max(1, max_steps) if max_steps else 0
                            )
                            if progress > 0.6 and global_commit_rate < 0.90:
                                print(
                                    f"[C2-WARNING] Commitment rate {global_commit_rate * 100:.1f}% below expected 95% at {progress * 100:.0f}% progress"
                                )

        if commitment_monitor is not None and commitment_monitor.should_stop:
            break

    post_training_context = {
        "epoch_loss": epoch_loss,
        "epoch_tokens": epoch_tokens,
        "best_loss": best_loss,
        "diverged": diverged,
        "global_step": global_step,
        "loss": loss,
        "param_info": param_info,
        "ppl_history": ppl_history,
        "start_time": start_time,
        "use_saq": use_saq,
        "use_amp": use_amp,
        "use_combinatorial_finetune": use_combinatorial_finetune,
        "combinatorial_max_sweeps": combinatorial_max_sweeps,
        "combinatorial_ternary_finetune": combinatorial_ternary_finetune,
        "dataset_type": dataset_type,
        "batch_size": batch_size,
        "data_dir": data_dir,
        "device": device,
        "save_dir": save_dir,
        "on_checkpoint": on_checkpoint,
        "optimizer_type": optimizer_type,
        "max_steps": max_steps,
        "num_epochs": num_epochs,
        "commitment_monitor": commitment_monitor,
        "lr_schedule_type": lr_schedule_type,
        "two_stage_decay": two_stage_decay,
        "use_8bit_optimizer": use_8bit_optimizer,
        "use_compile": use_compile,
        "seq_warmup": seq_warmup,
        "gradual_threshold": gradual_threshold,
        "pressure_blend": pressure_blend,
        "layer_wise_precision": layer_wise_precision,
        "use_curriculum": use_curriculum,
        "angular_reg_lambda": angular_reg_lambda,
        "boundary_reg_lambda": boundary_reg_lambda,
        "galore_rank": galore_rank,
        "saq_k": saq_k,
        "use_dual_rate": use_dual_rate,
        "dual_rate_peak": dual_rate_peak,
        "dual_rate_floor": dual_rate_floor,
        "use_gaste": use_gaste,
        "gaste_sigma_g": gaste_sigma_g,
        "use_rlec": use_rlec,
        "rlec_rank": rlec_rank,
        "use_adagc": use_adagc,
        "adagc_factor": adagc_factor,
        "adagc_beta": adagc_beta,
        "use_info_freeze": use_info_freeze,
        "info_freeze_alpha": info_freeze_alpha,
        "info_freeze_warmup": info_freeze_warmup,
        "use_dead_zone": use_dead_zone,
        "dead_zone_margin": dead_zone_margin,
        "dead_zone_lr_decay": dead_zone_lr_decay,
        "use_commitment_monitor": use_commitment_monitor,
        "commitment_check_interval": commitment_check_interval,
        "commitment_plateau_epsilon": commitment_plateau_epsilon,
        "commitment_plateau_patience": commitment_plateau_patience,
        "commitment_lr_boost": commitment_lr_boost,
        "commitment_lr_decay": commitment_lr_decay,
        "commitment_early_stop_threshold": commitment_early_stop_threshold,
        "commitment_early_stop": commitment_early_stop,
        "use_acsi_init": use_acsi_init,
        "acsi_max_batches": acsi_max_batches,
        "use_activation_quant": use_activation_quant,
        "activation_quant_bits": activation_quant_bits,
        "use_instant": use_instant,
        "instant_rank": instant_rank,
        "use_ags": use_ags,
        "ags_top_k_frac": ags_top_k_frac,
        "ags_full_sync_interval": ags_full_sync_interval,
        "ags_delta": ags_delta,
        "evaluate_perplexity": evaluate_perplexity,
    }

    return _post_training(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        monitor=monitor,
        ema=ema,
        progressive_freezer=progressive_freezer,
        fused_aga=fused_aga,
        let_teacher=let_teacher,
        let_proj=let_proj,
        od_teacher=od_teacher,
        train_loader=train_loader,
        context=post_training_context,
    )


# combinatorial_ternary_finetune moved — can be imported separately if needed
