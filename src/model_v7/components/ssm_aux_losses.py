"""SSM Auxiliary Losses — Batch H.

H1: Predictive SSM State Coding (MSE, lambda=0.01)
    The SSM output at position t predicts the residual stream value at t+1.
    Forces the hidden state to encode upcoming-token information.

H2: InfoNCE Mutual Information (cosine NCE, K=4 negatives, tau=0.07, lambda=0.005)
    Maximises MI between SSM state at t and next-token embedding at t+1.
    Positive pair: (state_t, x_{t+1}). Negatives: (state_t, x_{t+j}) j in [2,K+1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ssm_state_pred_loss(
    ssm_outputs: list,
    x_emb: torch.Tensor,
    pred_proj: nn.Linear,
    lambda_state: float = 0.01,
) -> torch.Tensor:
    """H1: MSE between pred_proj(state_t) and x_{t+1}."""
    if not ssm_outputs:
        return torch.tensor(0.0, device=x_emb.device, dtype=x_emb.dtype)

    total = torch.tensor(0.0, device=x_emb.device, dtype=x_emb.dtype)
    target = x_emb[:, 1:, :].detach()  # (B, L-1, d_model)

    for s in ssm_outputs:
        s_t = s[:, :-1, :]
        pred = pred_proj(s_t)
        total = total + F.mse_loss(pred, target)

    return lambda_state * total / len(ssm_outputs)


def compute_ssm_infonce_loss(
    ssm_outputs: list,
    x_emb: torch.Tensor,
    K: int = 4,
    temperature: float = 0.07,
    lambda_nce: float = 0.005,
) -> torch.Tensor:
    """H2: Cosine InfoNCE maximising MI between state_t and x_{t+1}."""
    if not ssm_outputs:
        return torch.tensor(0.0, device=x_emb.device, dtype=x_emb.dtype)

    total = torch.tensor(0.0, device=x_emb.device, dtype=x_emb.dtype)
    L = x_emb.shape[1]

    for s in ssm_outputs:
        B, seq_len, d = s.shape
        L_eff = min(seq_len - 1, L - 1)
        if L_eff <= 0:
            continue

        q = F.normalize(s[:, :L_eff, :], dim=-1)
        k_pos = F.normalize(x_emb[:, 1:L_eff + 1, :].detach(), dim=-1)
        sim_pos = (q * k_pos).sum(-1, keepdim=True)  # (B, L_eff, 1)

        neg_sims = []
        for offset in range(2, K + 2):
            end = L_eff + offset
            if end > L:
                end = L
                q_trim = q[:, : end - offset, :]
                k_neg = F.normalize(x_emb[:, offset:end, :].detach(), dim=-1)
                sim = (q_trim * k_neg).sum(-1)
                pad_len = L_eff - sim.shape[1]
                if pad_len > 0:
                    sim = F.pad(sim, (0, pad_len), value=0.0)
            else:
                k_neg = F.normalize(x_emb[:, offset:end, :].detach(), dim=-1)
                sim = (q * k_neg).sum(-1)
            neg_sims.append(sim.unsqueeze(-1))

        logits = torch.cat([sim_pos] + neg_sims, dim=-1) / temperature  # (B, L_eff, K+1)
        labels = torch.zeros(B * L_eff, dtype=torch.long, device=s.device)
        loss_layer = F.cross_entropy(logits.view(B * L_eff, K + 1), labels)
        total = total + loss_layer

    return lambda_nce * total / len(ssm_outputs)
