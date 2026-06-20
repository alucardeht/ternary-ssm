import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import contextlib
from pathlib import Path
from typing import Optional
from .model import ModelV2
from .config import ModelConfig
from .data import get_dataloader
from .components.router import MoDRouter
from .components.optim import CPUOffloadOptimizer

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TrainingMonitor:
    """Tracks training metrics for analysis."""

    def __init__(self):
        self.history = {
            'step': [], 'loss': [], 'lr': [], 'tokens_per_sec': [],
            'grad_norm': [], 'epoch': [],
        }
        self.router_history = {}
        self.ternary_history = {}

    def log(self, step: int, **kwargs):
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def print_status(self, step: int, loss: float, lr: float,
                     tokens_per_sec: float, grad_norm: float, extra: str = ''):
        print(f"Step {step:6d} | Loss {loss:.4f} | LR {lr:.2e} | "
              f"Tok/s {tokens_per_sec:.0f} | GradNorm {grad_norm:.4f}{extra}")


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def distillation_loss(student_logits, teacher_logits, temperature=2.0, use_tsld=False):
    """Compute knowledge distillation loss.

    Args:
        student_logits: (batch, seq_len, vocab_size) student model output
        teacher_logits: (batch, seq_len, vocab_size) teacher model output (detached)
        temperature: softening temperature for KL divergence
        use_tsld: if True, use Token-Scaled Logit Distillation

    Returns:
        kd_loss: scalar distillation loss
    """
    B, L, V = student_logits.shape
    s_logits = student_logits.view(B * L, V)
    t_logits = teacher_logits.view(B * L, V)

    s_log_probs = F.log_softmax(s_logits / temperature, dim=-1)
    t_probs = F.softmax(t_logits / temperature, dim=-1)

    if use_tsld:
        with torch.no_grad():
            teacher_confidence = t_probs.max(dim=-1).values
            token_scale = 1.0 - teacher_confidence
            token_scale = token_scale / (token_scale.mean() + 1e-8)

        kl_per_token = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=-1)
        kd_loss = (token_scale * kl_per_token).mean() * (temperature ** 2)
    else:
        kd_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature ** 2)

    return kd_loss


def evaluate_perplexity(
    model: nn.Module,
    config: ModelConfig,
    split: str = 'val',
    batch_size: int = 16,
    device: str = 'cpu',
    data_dir: str = None,
    use_amp: bool = False,
) -> float:
    """Evaluate perplexity on WikiText-103 validation or test set.

    Evaluates on non-overlapping chunks from WikiTextDataset.
    Perplexity = exp(average cross-entropy loss).
    """
    from .data import get_dataloader, WikiTextDataset

    dataset = WikiTextDataset(split=split, seq_len=config.seq_len, data_dir=data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp and str(device) == 'cuda'):
                logits = model(input_ids)

            loss = criterion(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))

    model.train()
    return perplexity


class ProgressiveScheduler:
    """Progressive training: FP16 warmup → Ternary QAT → Routing → Full.

    Based on 2025-2026 QAT research:
    - Phase 1 (FP16): Learn representations without quantization noise
    - Phase 2 (QAT): Enable ternary, let STE adapt weights
    - Phase 3 (Full): Enable routing on top of stable ternary
    """

    def __init__(self, model, total_steps, qat_start_pct=0.15, routing_start_pct=0.50, layer_wise=False):
        self.model = model
        self.total_steps = total_steps
        self.qat_start_step = int(total_steps * qat_start_pct)
        self.routing_start_step = int(total_steps * routing_start_pct)
        self.current_phase = 0
        self.layer_wise = layer_wise

        model.set_quantization_enabled(False)
        model.set_routing_enabled(False)
        model.reinit_ternary_standard()

    def step(self, global_step):
        """Call every training step. Returns phase name if phase changed, else None."""
        if self.current_phase == 0 and global_step >= self.qat_start_step:
            if self.layer_wise:
                keep_layers = [0, len(self.model.layers) - 1]
                self.model.set_quantization_layer_wise(True, keep_fp16_layers=keep_layers)
                msg = 'QAT enabled (layer-wise)'
            else:
                self.model.set_quantization_enabled(True)
                msg = 'QAT enabled'
            self.current_phase = 1
            return msg
        elif self.current_phase == 1 and global_step >= self.routing_start_step:
            self.model.set_routing_enabled(True)
            self.current_phase = 2
            return 'Routing enabled'
        return None

    @property
    def phase_name(self):
        return ['FP16 warmup', 'Ternary QAT', 'Full (QAT+Routing)'][self.current_phase]


def train_model(
    config: ModelConfig,
    dataset_type: str = 'synthetic',
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
    device: str = 'cpu',
    progressive: bool = False,
    qat_start_pct: float = 0.15,
    routing_start_pct: float = 0.50,
    lr_schedule_type: str = 'cosine',
    wsd_warmup_pct: float = 0.10,
    wsd_stable_pct: float = 0.60,
    two_stage_decay: bool = False,
    optimizer_type: str = 'adamw',
    use_8bit_optimizer: bool = False,
    shadow_bf16: bool = False,
    use_compile: bool = False,
    seq_warmup: bool = False,
    seq_warmup_min: int = 64,
    seq_warmup_pct: float = 0.30,
    ste_type: str = 'standard',
    gradual_threshold: bool = False,
    pressure_blend: bool = False,
    layer_wise_precision: bool = False,
    use_amp: bool = False,
    use_bf16: bool = False,
    use_gradient_checkpointing: bool = False,
    checkpoint_every_n: int = 1,
    use_sdpa: bool = False,
    use_cpu_offload: bool = False,
    use_distillation: bool = False,
    teacher_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    distill_alpha: float = 0.5,
    distill_temperature: float = 2.0,
    use_tsld: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> dict:
    """Train a Model v2 instance.

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
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config.use_amp = use_amp
    config.use_bf16 = use_bf16
    config.use_gradient_checkpointing = use_gradient_checkpointing
    config.checkpoint_every_n = checkpoint_every_n
    config.use_sdpa = use_sdpa
    config.use_torch_compile = use_compile
    config.use_cpu_offload = use_cpu_offload

    # Create model
    model = ModelV2(config).to(device)

    if shadow_bf16 and config.use_ternary:
        n_converted = model.convert_ternary_to_bf16()
        print(f"BF16 shadow weights: {n_converted} ternary layers converted")

    param_info = model.count_parameters()

    # torch.compile (PyTorch 2.0+, ~33% speedup with 80s warmup cost)
    if use_compile and device == 'cuda':
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Applying torch.compile (mode='default', dynamic=False)...")
        try:
            model = torch.compile(model, mode='default', dynamic=False)
            logger.info("torch.compile applied. First ~20 steps will be slower (compilation warmup).")
        except Exception as e:
            print(f"NOTE: torch.compile unavailable ({type(e).__name__}). Running without compilation.")
            print(f"  (torch.compile requires PyTorch 2.0+ with Triton support.)")

    print(f"\n{'='*60}")
    print(f"Model v2 Training")
    print(f"{'='*60}")
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Ternary: {param_info['ternary']:,} ({param_info['ternary_pct']:.1f}%)")
    print(f"FP16: {param_info['fp16']:,}")
    print(f"Memory (FP16): {param_info['memory_mb_fp16']:.1f} MB")
    print(f"Memory (mixed): {param_info['memory_mb_mixed']:.1f} MB")
    print(f"Device: {device}")
    print(f"SSM layers: {sum(1 for i in range(config.n_layer) if not config.is_swa_layer(i))}")
    print(f"SWA layers: {sum(1 for i in range(config.n_layer) if config.is_swa_layer(i))}")
    print(f"Ternary: {config.use_ternary}")
    print(f"Routing: {config.use_routing}")
    print(f"Progressive training: {progressive}")
    if progressive:
        print(f"  - QAT starts at {qat_start_pct*100:.0f}% of training")
        print(f"  - Routing starts at {routing_start_pct*100:.0f}% of training")
    print(f"Grad accumulation: {grad_accum_steps} steps (effective batch = {batch_size * grad_accum_steps})")
    print(f"LR schedule: {lr_schedule_type}")
    if lr_schedule_type == 'wsd':
        print(f"  WSD: {wsd_warmup_pct*100:.0f}% warmup, {wsd_stable_pct*100:.0f}% stable, {(1-wsd_warmup_pct-wsd_stable_pct)*100:.0f}% decay")
    print(f"Two-stage weight decay: {two_stage_decay}")
    print(f"8-bit optimizer: {use_8bit_optimizer}")
    if shadow_bf16:
        print(f"BF16 shadow weights: enabled (ternary layers only)")
    if optimizer_type != 'adamw':
        print(f"Optimizer type: {optimizer_type}")
    print(f"torch.compile: {use_compile}")
    if ste_type != 'standard':
        print(f"STE type: {ste_type}")
    if gradual_threshold:
        print(f"Gradual threshold: 0.5 -> 0.7")
    if pressure_blend:
        print(f"Pressure blend: enabled (20% QAT duration)")
    if layer_wise_precision:
        print(f"Layer-wise precision: first/last layers FP16")
    if seq_warmup:
        print(f"Sequence warmup: {seq_warmup_min} -> {config.seq_len} over {seq_warmup_pct*100:.0f}% of training")
    if use_amp:
        print(f"AMP (Automatic Mixed Precision): enabled")
    if use_gradient_checkpointing:
        print(f"Gradient checkpointing: enabled (every {checkpoint_every_n} layer(s))")
    if use_sdpa:
        print(f"SDPA (scaled_dot_product_attention): enabled")
    print(f"Memory limit: {max_memory_mb} MB")
    print(f"{'='*60}\n")

    # Data
    train_loader = get_dataloader(
        dataset_type, config.vocab_size, config.seq_len, batch_size, num_samples,
        data_dir=data_dir, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Optimizer with differential LRs
    param_groups = model.get_param_groups(base_lr)

    if optimizer_type == 'muon':
        from src.model_v2.components.optim import Muon as MuonOpt
        muon_groups = model.get_muon_param_groups(base_lr)
        muon_count = sum(p.numel() for p in muon_groups[0]['params'])
        adamw_count = sum(p.numel() for p in muon_groups[1]['params'])
        optimizer = MuonOpt(muon_groups, lr=base_lr * 10, momentum=0.95,
                            ns_steps=5, weight_decay=0.01)
        print(f"Using Muon optimizer ({muon_count:,} Muon params, {adamw_count:,} AdamW params)")

    elif optimizer_type == 'schedule_free':
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(param_groups, lr=base_lr, betas=(0.9, 0.95),
                                       weight_decay=0.1, warmup_steps=warmup_steps)
        print(f"Using Schedule-Free AdamW (warmup={warmup_steps} steps)")

    elif optimizer_type == 'ademamix':
        from pytorch_optimizer import AdEMAMix
        optimizer = AdEMAMix(param_groups, lr=base_lr, betas=(0.9, 0.999, 0.9999),
                             weight_decay=0.1, alpha=5.0)
        print(f"Using AdEMAMix optimizer (alpha=5.0, beta3=0.9999)")

    elif optimizer_type == 'spam':
        from pytorch_optimizer import SPAM
        optimizer = SPAM(param_groups, lr=base_lr, weight_decay=0.1)
        print(f"Using SPAM optimizer")

    elif use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
            print(f"Using 8-bit AdamW optimizer (bitsandbytes)")
        except ImportError:
            print(f"WARNING: bitsandbytes not available, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)

    # Load teacher model for distillation
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
            print(f"WARNING: Teacher vocab ({teacher_vocab}) != Student vocab ({config.vocab_size})")
        print(f"Teacher loaded. VRAM after: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # AMP setup
    scaler = torch.amp.GradScaler('cuda') if (use_amp or use_bf16) and device == 'cuda' else None
    if use_bf16:
        scaler = None

    # LR scheduler: linear warmup + cosine decay
    total_steps = max_steps or (num_epochs * len(train_loader))

    def lr_schedule(step):
        if lr_schedule_type == 'wsd':
            # Warmup-Stable-Decay (Alibaba Oct 2024)
            warmup_end = int(total_steps * wsd_warmup_pct)
            stable_end = int(total_steps * (wsd_warmup_pct + wsd_stable_pct))
            if step < warmup_end:
                return step / max(1, warmup_end)
            elif step < stable_end:
                return 1.0
            else:
                decay_progress = (step - stable_end) / max(1, total_steps - stable_end)
                return max(0.0, 1.0 - decay_progress)
        else:
            # Cosine (default)
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    if optimizer_type == 'schedule_free':
        scheduler = None  # Schedule-Free handles its own scheduling
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if use_cpu_offload and device == 'cuda':
        optimizer = CPUOffloadOptimizer(optimizer, device=device)
        print(f"CPU offload: optimizer states -> pinned CPU RAM")

    # Progressive scheduler (optional)
    progressive_scheduler = None
    if progressive and (config.use_ternary or config.use_routing):
        total_optimizer_steps = total_steps // grad_accum_steps
        progressive_scheduler = ProgressiveScheduler(model, total_optimizer_steps, qat_start_pct, routing_start_pct, layer_wise=layer_wise_precision)
        print(f"Progressive scheduler ({total_optimizer_steps} optimizer steps):")
        print(f"  FP16 -> QAT@step {progressive_scheduler.qat_start_step} -> Routing@step {progressive_scheduler.routing_start_step}")

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Monitoring
    monitor = TrainingMonitor()

    # Set gradient estimator type for ternary layers
    if ste_type != 'standard' and config.use_ternary:
        model.set_ste_type(ste_type)
        if ste_type == 'hestia':
            model.set_training_progress(0, total_steps)

    if gradual_threshold:
        model.set_gradual_threshold(True, start=0.5, end=0.7)
        print(f"Gradual threshold: 0.5 -> 0.7 over training")

    if pressure_blend:
        model.set_pressure_blend(True, duration=0.2)
        print(f"Pressure blend: gradual FP->ternary over 20% of QAT")

    model.train()
    if optimizer_type == 'schedule_free':
        optimizer.train()
    global_step = 0
    best_loss = float('inf')
    start_time = time.time()
    loss = None

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_tokens = 0
        optimizer.zero_grad()

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            if max_steps and global_step >= max_steps:
                break

            mem_mb = 0
            if HAS_PSUTIL and batch_idx % 50 == 0:
                mem_mb = psutil.Process().memory_info().rss / 1024**2
                if mem_mb > max_memory_mb:
                    print(f"\n[MEMORY GUARD] RSS = {mem_mb:.0f} MB > {max_memory_mb} MB limit.")
                    raise MemoryError(f"Training exceeded {max_memory_mb} MB limit ({mem_mb:.0f} MB used).")

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            if seq_warmup:
                warmup_end_step = int(total_steps * seq_warmup_pct)
                if global_step < warmup_end_step:
                    progress = global_step / max(1, warmup_end_step)
                    current_seq_len = int(seq_warmup_min + (config.seq_len - seq_warmup_min) * progress)
                    current_seq_len = max(seq_warmup_min, (current_seq_len // 8) * 8)
                    if current_seq_len < input_ids.shape[1]:
                        input_ids = input_ids[:, :current_seq_len].contiguous()
                        targets = targets[:, :current_seq_len].contiguous()

            if use_bf16 and device == 'cuda':
                amp_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            elif use_amp and device == 'cuda':
                amp_context = torch.amp.autocast(device_type='cuda')
            else:
                amp_context = contextlib.nullcontext()

            with amp_context:
                logits = model(input_ids)
                ce_loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_model(input_ids).logits
                    kd_loss = distillation_loss(
                        logits, teacher_logits,
                        temperature=distill_temperature,
                        use_tsld=use_tsld,
                    )
                    loss = (1 - distill_alpha) * ce_loss + distill_alpha * kd_loss
                else:
                    loss = ce_loss
            scaled_loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            batch_tokens = input_ids.numel()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens

            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                # Two-stage weight decay (BitNet recipe: 0.01 first half, 0.0 second half)
                if two_stage_decay and global_step >= total_steps // 2:
                    for pg in optimizer.param_groups:
                        if pg.get('name') != 'embedding':  # embedding already has wd=0
                            pg['weight_decay'] = 0.0

                optimizer.zero_grad()
                global_step += 1

                # Update training progress for HESTIA temperature scheduling
                if ste_type == 'hestia' and config.use_ternary:
                    model.set_training_progress(global_step, total_steps)

                if progressive_scheduler is not None:
                    phase_msg = progressive_scheduler.step(global_step)
                    if phase_msg:
                        print(f"\n[PROGRESSIVE] Step {global_step}: {phase_msg}\n")

                if global_step % log_every == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                    try:
                        current_lr = optimizer.param_groups[0]['lr']
                    except (IndexError, KeyError):
                        current_lr = base_lr

                    log_dict = {
                        'step': global_step, 'loss': loss.item(), 'lr': current_lr,
                        'tokens_per_sec': tokens_per_sec,
                        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        'epoch': epoch,
                    }
                    if progressive_scheduler is not None:
                        log_dict['phase'] = progressive_scheduler.phase_name
                    monitor.log(**log_dict)

                    extra = f' | Mem {mem_mb:.0f}MB'
                    if teacher_model is not None:
                        extra += f' | CE {ce_loss.item():.4f} KD {kd_loss.item():.4f}'
                    if progressive_scheduler is not None:
                        extra += f' | Phase {progressive_scheduler.phase_name}'
                    if config.use_ternary:
                        for name, module in model.named_modules():
                            if hasattr(module, 'get_ternary_stats'):
                                stats = module.get_ternary_stats()
                                extra += f" | Tern +{stats['positive']:.2f} 0:{stats['zero']:.2f} -{stats['negative']:.2f}"
                                break

                    if config.use_routing:
                        routing_stats = []
                        for name, module in model.named_modules():
                            if hasattr(module, 'get_stats') and isinstance(module, MoDRouter):
                                stats = module.get_stats()
                                routing_stats.append(stats)
                        if routing_stats:
                            avg_ratio = sum(s['process_ratio'] for s in routing_stats) / len(routing_stats)
                            avg_weight = sum(s['mean_weight'] for s in routing_stats) / len(routing_stats)
                            extra += f" | Route {avg_ratio:.2f} MeanW {avg_weight:.3f}"

                    monitor.print_status(global_step, loss.item(), current_lr, tokens_per_sec,
                                       grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, extra)

        avg_loss = epoch_loss / max(epoch_tokens, 1)
        print(f"\n--- Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} ---\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

        if max_steps and global_step >= max_steps:
            break

    total_time = time.time() - start_time

    # Peak VRAM
    peak_vram_mb = 0
    if device == 'cuda' and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    results = {
        'total_steps': global_step,
        'best_loss': best_loss,
        'final_loss': loss.item() if loss is not None else float('inf'),
        'total_time_sec': total_time,
        'tokens_per_sec': epoch_tokens / total_time if total_time > 0 else 0,
        'peak_vram_mb': peak_vram_mb,
        'param_count': param_info,
        'config': {
            'lr_schedule': lr_schedule_type,
            'two_stage_decay': two_stage_decay,
            'optimizer_type': optimizer_type,
            'use_8bit_optimizer': use_8bit_optimizer,
            'use_compile': use_compile,
            'seq_warmup': seq_warmup,
            'gradual_threshold': gradual_threshold,
            'pressure_blend': pressure_blend,
            'layer_wise_precision': layer_wise_precision,
        },
    }

    # Save
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        monitor.save(f"{save_dir}/training_log.json")
        torch.save(model.state_dict(), f"{save_dir}/final_model.pt")
        with open(f"{save_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {save_dir}/")

    if optimizer_type == 'schedule_free':
        optimizer.eval()

    if dataset_type == 'wikitext':
        print(f"\nEvaluating perplexity on validation set...")
        val_ppl = evaluate_perplexity(model, config, split='val',
                                       batch_size=batch_size, device=device, use_amp=use_amp,
                                       data_dir=data_dir)
        results['val_perplexity'] = val_ppl
        print(f"Validation perplexity: {val_ppl:.2f}")

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Steps: {global_step}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Time: {total_time:.1f}s")
    print(f"{'='*60}\n")

    return results
