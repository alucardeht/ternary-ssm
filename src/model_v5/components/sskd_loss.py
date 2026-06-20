"""Stepwise Self-Knowledge Distillation (R4-A3-F4).

Double forward: shadow weights produce teacher logits,
quantized weights produce student logits. KL divergence loss.
"""
import torch
import torch.nn.functional as F


def compute_sskd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL divergence between shadow-weight and quantized-weight outputs.

    Args:
        student_logits: (batch*seq, vocab) from quantized forward
        teacher_logits: (batch*seq, vocab) from shadow forward
        temperature: softmax temperature (higher = softer distributions)

    Returns:
        Scalar KL divergence loss.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl * (temperature ** 2)
