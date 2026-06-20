import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import contextlib
import argparse
from pathlib import Path
from typing import Optional, Dict
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from .model import ModelV2
from .config import config_wikitext_50m, ModelConfig
from .data import get_dataloader
from .components.optim import AdamWBF16
from .train import distillation_loss, evaluate_perplexity, compute_grad_norm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def distill(
    teacher_path: str,
    student_path: str,
    data_dir: str,
    device: str = 'cuda',
    num_epochs: int = 3,
    batch_size: int = 32,
    grad_accum_steps: int = 1,
    lr: float = 2e-5,
    warmup_steps: int = 50,
    max_steps: Optional[int] = None,
    temperature: float = 10.0,
    alpha: float = 0.5,
    use_tsld: bool = True,
    log_every: int = 50,
    eval_every: int = 500,
    save_dir: Optional[str] = None,
    max_memory_mb: int = 10000,
    use_gradient_checkpointing: bool = True,
    use_bf16: bool = True,
) -> Dict:
    """Distill teacher knowledge into student model.

    Args:
        teacher_path: Path to teacher checkpoint
        student_path: Path to student checkpoint
        data_dir: Path to WikiText training data
        device: Device ('cuda' or 'cpu')
        num_epochs: Number training epochs
        batch_size: Training batch size
        grad_accum_steps: Gradient accumulation steps
        lr: Learning rate
        warmup_steps: Warmup steps for learning rate scheduler
        max_steps: Maximum training steps (None = no limit)
        temperature: Temperature for knowledge distillation
        alpha: Weight between CE loss (1-alpha) and KD loss (alpha)
        use_tsld: Use Token-Scaled Logit Distillation
        log_every: Log metrics every N steps
        eval_every: Evaluate every N steps
        save_dir: Directory to save best checkpoint (None = no save)
        max_memory_mb: Max GPU memory to use (safety limit)
        use_gradient_checkpointing: Enable gradient checkpointing on student
        use_bf16: Use BF16 autocast for training

    Returns:
        Dict with training metrics and results
    """
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading teacher from {teacher_path}")
    teacher_config = config_wikitext_50m()
    teacher_config.use_ternary = False
    teacher_config.use_routing = False
    teacher_config.use_ssd = True
    teacher_config.use_torch_compile = False
    teacher = ModelV2(teacher_config)
    ck = torch.load(teacher_path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        teacher_sd = ck['model_state_dict']
    elif isinstance(ck, dict) and 'model_state' in ck:
        teacher_sd = ck['model_state']
    elif isinstance(ck, dict) and any(k.startswith('embedding') or k.startswith('layers') for k in ck.keys()):
        teacher_sd = ck
    else:
        teacher_sd = ck
    missing, unexpected = teacher.load_state_dict(teacher_sd, strict=False)
    print(f"Teacher loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

    print(f"Loading student from {student_path}")
    student_config = config_wikitext_50m()
    student_config.use_ternary = True
    student_config.use_routing = True
    student_config.use_ssd = True
    student_config.use_torch_compile = False
    if use_gradient_checkpointing:
        student_config.use_gradient_checkpointing = True
        student_config.checkpoint_every_n = 1

    student = ModelV2(student_config)
    ck = torch.load(student_path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        student_sd = ck['model_state_dict']
    elif isinstance(ck, dict) and 'model_state' in ck:
        student_sd = ck['model_state']
    elif isinstance(ck, dict) and any(k.startswith('embedding') or k.startswith('layers') for k in ck.keys()):
        student_sd = ck
    else:
        student_sd = ck
    missing, unexpected = student.load_state_dict(student_sd, strict=False)
    print(f"Student loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    student = student.to(device)
    student.convert_ternary_to_bf16()
    student.train()
    print(f"Student loaded: {sum(p.numel() for p in student.parameters()):,} params")

    data_loader = get_dataloader(
        dataset_type='wikitext',
        vocab_size=student_config.vocab_size,
        seq_len=student_config.seq_len,
        batch_size=batch_size,
        split='train',
        data_dir=data_dir,
        num_workers=0,
    )

    param_groups = student.get_param_groups(lr)
    optimizer = AdamWBF16(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

    total_steps_estimate = min(
        len(data_loader) * num_epochs,
        max_steps if max_steps is not None else len(data_loader) * num_epochs,
    )

    def warmup_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps_estimate - warmup_steps))

    scheduler = LambdaLR(optimizer, warmup_schedule)

    criterion = nn.CrossEntropyLoss(reduction='mean')

    print("\nEvaluating baseline models...")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        initial_teacher_ppl = evaluate_perplexity(
            teacher,
            teacher_config,
            split='val',
            batch_size=batch_size,
            device=device,
            data_dir=data_dir,
            use_amp=False,
        )
    print(f"Teacher (BF16) PPL: {initial_teacher_ppl:.2f}")

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        initial_student_ppl = evaluate_perplexity(
            student,
            student_config,
            split='val',
            batch_size=batch_size,
            device=device,
            data_dir=data_dir,
            use_amp=False,
        )
    print(f"Student (before distill) PPL: {initial_student_ppl:.2f}")

    print(f"\nStarting distillation | T={temperature} alpha={alpha} TSLD={use_tsld}")
    print(f"LR={lr} warmup={warmup_steps} epochs={num_epochs}")

    start_time = time.time()
    global_step = 0
    accum_loss = 0.0
    accum_ce_loss = 0.0
    accum_kd_loss = 0.0
    best_val_ppl = initial_student_ppl
    best_step = 0

    amp_context = (
        torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        if use_bf16 and str(device) == 'cuda'
        else contextlib.nullcontext()
    )

    for epoch in range(num_epochs):
        epoch_start = time.time()
        student.train()

        for batch_idx, (input_ids, targets) in enumerate(data_loader):
            if max_steps is not None and global_step >= max_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            with amp_context:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    student_logits = student(input_ids)

                ce_loss = criterion(
                    student_logits.view(-1, student_config.vocab_size),
                    targets.view(-1),
                )
                kd_loss = distillation_loss(
                    student_logits,
                    teacher_logits.float(),
                    temperature=temperature,
                    use_tsld=use_tsld,
                )
                loss = (1 - alpha) * ce_loss + alpha * kd_loss

            loss = loss / grad_accum_steps
            loss.backward()

            accum_loss += loss.item()
            accum_ce_loss += ce_loss.item() / grad_accum_steps
            accum_kd_loss += kd_loss.item() / grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0:
                clip_grad_norm_(student.parameters(), 1.0)
                grad_norm = compute_grad_norm(student)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (global_step + 1) % log_every == 0:
                    avg_loss = accum_loss / log_every
                    avg_ce = accum_ce_loss / log_every
                    avg_kd = accum_kd_loss / log_every
                    current_lr = optimizer.param_groups[0]['lr']

                    if HAS_PSUTIL:
                        peak_vram = 0
                        try:
                            if str(device) == 'cuda':
                                peak_vram = torch.cuda.max_memory_allocated() / 1024**2
                        except Exception:
                            pass
                    else:
                        peak_vram = 0

                    print(
                        f"Epoch {epoch+1} Step {global_step+1:6d} | "
                        f"Loss {avg_loss:.4f} (CE {avg_ce:.4f} KD {avg_kd:.4f}) | "
                        f"LR {current_lr:.2e} | GradNorm {grad_norm:.4f} | "
                        f"VRAM {peak_vram:.0f}MB"
                    )

                    accum_loss = 0.0
                    accum_ce_loss = 0.0
                    accum_kd_loss = 0.0

                if (global_step + 1) % eval_every == 0:
                    print(f"Evaluating at step {global_step + 1}...")
                    student.eval()
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        val_ppl = evaluate_perplexity(
                            student,
                            student_config,
                            split='val',
                            batch_size=batch_size,
                            device=device,
                            data_dir=data_dir,
                            use_amp=False,
                        )
                    student.train()

                    print(f"  Val PPL: {val_ppl:.2f}")

                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        best_step = global_step + 1

                        if save_dir is not None:
                            save_path = Path(save_dir) / f"student_best.pt"
                            torch.save(
                                {
                                    'model_state': student.state_dict(),
                                    'config': student_config,
                                    'step': global_step + 1,
                                    'val_ppl': val_ppl,
                                },
                                save_path,
                            )
                            print(f"  Saved best checkpoint to {save_path}")

                global_step += 1

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")

        if max_steps is not None and global_step >= max_steps:
            break

    training_time = time.time() - start_time
    final_loss = accum_loss

    print("\nFinal evaluation...")
    student.eval()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        final_student_ppl = evaluate_perplexity(
            student,
            student_config,
            split='val',
            batch_size=batch_size,
            device=device,
            data_dir=data_dir,
            use_amp=False,
        )
    student.train()

    peak_vram_mb = 0
    if HAS_PSUTIL and str(device) == 'cuda':
        try:
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            pass

    ppl_gap_closed = max(
        0.0,
        100.0 * (initial_student_ppl - final_student_ppl) / (initial_student_ppl - initial_teacher_ppl)
        if initial_student_ppl > initial_teacher_ppl
        else 0.0,
    )

    result_dict = {
        'initial_teacher_ppl': initial_teacher_ppl,
        'initial_student_ppl': initial_student_ppl,
        'final_student_ppl': final_student_ppl,
        'best_student_ppl': best_val_ppl,
        'ppl_gap_closed_pct': ppl_gap_closed,
        'final_loss': final_loss,
        'best_step': best_step,
        'total_steps': global_step,
        'training_time': training_time,
        'peak_vram_mb': peak_vram_mb,
        'temperature': temperature,
        'alpha': alpha,
        'use_tsld': use_tsld,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'warmup_steps': warmup_steps,
    }

    print("\n" + "="*80)
    print("DISTILLATION SUMMARY")
    print("="*80)
    print(f"Teacher PPL (FP16):        {initial_teacher_ppl:.2f}")
    print(f"Student PPL Before:        {initial_student_ppl:.2f}")
    print(f"Student PPL After:         {final_student_ppl:.2f}")
    print(f"Best Student PPL:          {best_val_ppl:.2f} (step {best_step})")
    print(f"Gap Closed:                {ppl_gap_closed:.1f}%")
    print(f"Steps:                     {global_step}")
    print(f"Training Time:             {training_time/3600:.2f}h")
    print(f"Peak VRAM:                 {peak_vram_mb:.0f}MB")
    print("="*80)

    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distill teacher knowledge into student model')
    parser.add_argument('--teacher-path', type=str, required=True, help='Path to teacher checkpoint')
    parser.add_argument('--student-path', type=str, required=True, help='Path to student checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to WikiText training data')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--use-tsld', action='store_true', default=True)
    parser.add_argument('--no-tsld', dest='use_tsld', action='store_false')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--max-memory-mb', type=int, default=10000)
    parser.add_argument('--no-gradient-checkpointing', dest='use_gradient_checkpointing',
                       action='store_false', default=True)
    parser.add_argument('--no-bf16', dest='use_bf16', action='store_false', default=True)

    args = parser.parse_args()

    result = distill(
        teacher_path=args.teacher_path,
        student_path=args.student_path,
        data_dir=args.data_dir,
        device=args.device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        temperature=args.temperature,
        alpha=args.alpha,
        use_tsld=args.use_tsld,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        max_memory_mb=args.max_memory_mb,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_bf16=args.use_bf16,
    )

    print(f"\nResults saved to memory. Final PPL: {result['final_student_ppl']:.2f}")
