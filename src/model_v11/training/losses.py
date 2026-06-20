import torch
import torch.nn as nn
import torch.nn.functional as F


def margin_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Hinge/margin loss on raw logits. No softmax/exp/log."""
    correct_logits = logits[torch.arange(logits.size(0), device=logits.device), targets]
    logits_wrong = logits.clone()
    logits_wrong[torch.arange(logits.size(0), device=logits.device), targets] = float('-inf')
    max_wrong = logits_wrong.max(dim=-1).values
    return torch.clamp(margin - correct_logits + max_wrong, min=0.0).mean()


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

        kl_per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
        kd_loss = (token_scale * kl_per_token).mean() * (temperature**2)
    else:
        kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (
            temperature**2
        )

    return kd_loss
