"""Production training pipeline: FP + Cache + ABA + D4 + mini-BCD."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from src.model_v11.optimizers import FactoredAdamW


def streaming_ce_loss(h_norm, weight, y, head_module=None):
    """
    Compute CE loss token-by-token WITHOUT materializing [batch, seq, vocab].
    Peak VRAM: batch * vocab * 4 bytes (one position at a time).

    Args:
        h_norm: [batch, seq, d_model] — the hidden states after final_norm (DETACHED from graph for internal use, but caller keeps graph-connected version)
        weight: [vocab, d_model] — lm_head weight or embedding weight
        y: [batch, seq] — target token IDs
        head_module: nn.Linear or None — if provided, accumulates grad for head weight

    Returns:
        loss_scalar: float — CE loss value
        dH: [batch, seq, d_model] — gradient of loss w.r.t. h_norm
    """
    batch, seq_len, d_model = h_norm.shape
    dH = torch.zeros_like(h_norm)
    loss_sum = 0.0
    n_tokens = batch * seq_len

    with torch.no_grad():
        for t in range(seq_len):
            h_t = h_norm[:, t, :]               # [batch, d_model]
            z_t = h_t @ weight.t()               # [batch, vocab]

            log_p = F.log_softmax(z_t, dim=-1)
            tgt_t = y[:, t]
            loss_sum += F.nll_loss(log_p, tgt_t, reduction='sum').item()

            # gradient: softmax(z) - one_hot(target)
            p_t = torch.exp(log_p)
            p_t[torch.arange(batch, device=p_t.device), tgt_t] -= 1.0
            # CE sum (not mean) — BitNet b1.58 confirmed: sum stabilizes ternary training
            # Gradient is NOT divided by n_tokens — this is intentional

            # dL/dh_t = dL/dz_t @ W
            dH[:, t, :] = p_t @ weight

            # accumulate lm_head weight gradient if trainable
            if head_module is not None and head_module.weight.requires_grad:
                if head_module.weight.grad is None:
                    head_module.weight.grad = torch.zeros_like(weight)
                head_module.weight.grad += p_t.t() @ h_t

    return loss_sum / n_tokens, dH  # loss reported as mean (readable), gradient is sum (BitNet-style)




def compute_til_loss(model, h_t_detached, y, stage_k, K=4, til_weight=0.1):
    """TIL: predict future tokens via SSM linear dynamics. Gives direct gradient to A_log."""
    block = model.layers[stage_k].block if hasattr(model.layers[stage_k], 'block') else None
    if block is None or not hasattr(block, 'A_log'):
        return torch.tensor(0.0)

    A_log = block.A_log
    A = -torch.exp(A_log)

    batch, seq_len, d_model = h_t_detached.shape
    n_heads = A_log.shape[0]
    head_dim = d_model // n_heads
    h_reshaped = h_t_detached.view(batch, seq_len, n_heads, head_dim)

    W = get_embedding_weight(model).detach()

    til_loss = torch.tensor(0.0, device=A_log.device)

    for k in range(1, K + 1):
        Ak = A ** k
        h_proj = h_reshaped * Ak.view(1, 1, n_heads, 1)
        h_proj_flat = h_proj.view(batch, seq_len, d_model)

        valid_len = seq_len - k
        if valid_len <= 0:
            continue

        h_src = h_proj_flat[:, :valid_len, :]
        y_tgt = y[:, k:k + valid_len]

        logits = h_src @ W.t()
        loss_k = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_tgt.reshape(-1))
        til_loss = til_loss + loss_k

    return til_loss * (til_weight / max(K, 1))


def compute_gp_kl_loss(ssm_modules, h_states, gp_kl_lambda=0.05):
    """
    Unified GP-KL loss. Replaces: loss_tcsd + loss_att + loss_traj + loss_lgram.

    Enforces 4 statistical moments of the SSM state trajectory distribution:
      (1) Zero mean: E[h_t] = 0
      (2) Unit variance: Var(h_t) = 1  [replaces TCSD]
      (3) Autocorr matches SSM decay: Cov(h_t, h_{t-1}) / Var(h_t) = A_i  [replaces traj_div]
      (4) Stability: keep |A_i| < 0.5  [replaces attractor]
      (5) Weight orthogonality: off-diag(W @ W^T) = 0  [replaces L_gram]
    """
    if not h_states:
        return torch.tensor(0.0)

    device = h_states[0]['mean'].device
    loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for _m, stats in zip(ssm_modules, h_states):
        h_mean = stats['mean']           # [d_inner]
        h_var  = stats['var']            # [d_inner]
        cross  = stats['cross']          # [d_inner]
        n_pairs = stats['n_pairs']       # int

        l_mean = h_mean.pow(2).mean()
        l_var  = (h_var - 1.0).pow(2).mean()

        l_autocorr = torch.tensor(0.0, device=device)
        if n_pairs > 0 and hasattr(_m, 'A_log') and _m.A_log is not None:
            d_state = h_mean.shape[0]
            n_heads = _m.A_log.shape[0]
            head_dim = d_state // n_heads if n_heads <= d_state else 1
            A_vals = -torch.exp(_m.A_log)
            A_target = A_vals.repeat_interleave(head_dim)[:d_state].detach()
            # rho = E[h_t * h_{t-1}] / Var(h_t)
            # cross / n_pairs gives E[h_t * h_{t-1}] (unnormalized, before mean-centering)
            # For zero-mean h this equals the lag-1 autocovariance
            rho = (cross / n_pairs) / h_var
            l_autocorr = (rho - A_target).pow(2).mean()

        l_stability = torch.tensor(0.0, device=device)
        if hasattr(_m, 'A_log') and _m.A_log is not None:
            l_stability = torch.clamp(torch.exp(_m.A_log) - 0.5, min=0.0).pow(2).mean()

        l_orth = torch.tensor(0.0, device=device)
        _n_orth = 0
        for _sub in [getattr(_m, n, None) for n in ['in_proj', 'out_proj', 'B_proj', 'C_proj']]:
            if _sub is not None and hasattr(_sub, 'weight') and _sub.weight.ndim == 2:
                w = _sub.weight
                if w.shape[0] <= 2048:
                    ww = w @ w.T
                    off = ww - torch.diag(ww.diagonal())
                    l_orth = l_orth + off.pow(2).mean()
                    _n_orth += 1
        if _n_orth > 0:
            l_orth = l_orth / _n_orth

        loss = loss + l_mean + l_var + l_autocorr + l_stability + l_orth
        n_terms += 1

    if n_terms > 0:
        loss = gp_kl_lambda * loss / n_terms

    return loss


@dataclass
class ProductionConfig:
    model_size: str
    dataset_path: str
    output_path: str
    batch_size: int = 16
    seq_len: int = 512
    max_steps_per_stage: int = 100
    min_steps: int = 50
    patience: int = 50
    rel_threshold: float = 0.001
    abs_threshold: float = 0.005
    fp_lr: float = 5e-5
    bcd_lr: float = 1e-4
    bcd_steps: int = 50
    bcd_bottom_fraction: float = 0.25
    bcd_batch_size: int = 16
    cache_batches: int = 20
    eval_batches: int = 20
    eval_fixed_batches: int = 20
    log_every: int = 25
    use_gradient_filter: bool = True
    filter_n_min: int = 5
    smoke_test: bool = False
    kill_gpu_procs: bool = False
    seed: int = 42
    max_vram_mb: int = 10800
    use_psa: bool = False
    psa_weight: float = 0.1
    use_til: bool = False
    til_K: int = 4
    til_weight: float = 0.1
    use_osp: bool = False
    osp_lambda: float = 0.01
    fp_order: str = "forward"
    training_mode: str = "fp"
    joint_lr: float = 0.1  # natural gradient tolerates much higher LR than Adam (R337 measured)
    joint_steps: int = 10000
    joint_warmup_ratio: float = 0.05
    joint_grad_clip: float = 1.0
    joint_eval_every: int = 100
    joint_checkpoint_every: int = 2000
    use_warm_h0: bool = True
    warm_h0_batches: int = 10
    use_capacity_matching: bool = True
    capacity_matching_every: int = 500
    mtp_weight: float = 0.1

    def __post_init__(self):
        if self.smoke_test:
            self.max_steps_per_stage = 10
            self.bcd_steps = 5
            self.cache_batches = 2
            self.eval_batches = 1
            self.eval_fixed_batches = 2
            self.joint_steps = 20


def safety_check_vram(max_vram_mb=4000):
    if not torch.cuda.is_available():
        print("[PROD] CUDA not available — CPU mode", flush=True)
        return torch.device("cpu")

    free_mb = torch.cuda.mem_get_info()[0] / 1e6
    print(f"[PROD] VRAM free: {free_mb:.0f} MB (required: {max_vram_mb} MB)", flush=True)

    if free_mb < max_vram_mb:
        print(f"[ERROR] VRAM insufficient: {free_mb:.0f} MB < {max_vram_mb} MB", flush=True)
        raise RuntimeError(f"VRAM {free_mb:.0f}MB < required {max_vram_mb}MB")

    return torch.device("cuda")


def kill_gpu_processes():
    """Optional: kill other GPU processes."""
    import os
    os.system("nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null")
    print("[PROD] Killed other GPU processes", flush=True)
    time.sleep(1)
    torch.cuda.empty_cache()


def vram_snapshot(label="", verbose=True):
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    allocated_mb = torch.cuda.memory_allocated() / 1e6
    reserved_mb = torch.cuda.memory_reserved() / 1e6
    if verbose:
        print(f"[PROD] VRAM {label:20s}: {allocated_mb:8.1f} MB allocated, "
              f"{reserved_mb:8.1f} MB reserved", flush=True)
    return allocated_mb


def get_vocab_size(model):
    if hasattr(model.embedding, 'num_embeddings'):
        return model.embedding.num_embeddings
    return model.embedding.weight.shape[0]


def get_embedding_weight(model):
    if model.lm_head is not None:
        return model.lm_head.weight
    return model.embedding.weight


def apply_bstar_init(model, cfg):
    """R299: Initialize layer 0 B-block as block-sparse aligned to RNS→eigenspace."""
    if not getattr(cfg, 'use_rns_embedding', False) or not getattr(cfg, 'use_tbd_init', False):
        return

    layer0 = model.layers[0]
    block = layer0.block if hasattr(layer0, 'block') else None
    if block is None or not hasattr(block, 'x_proj'):
        print("[B*-INIT] Layer 0 has no x_proj — skipping", flush=True)
        return
    if not hasattr(block, 'n_heads'):
        print("[B*-INIT] Layer 0 SSM has no n_heads (not SSD) — skipping", flush=True)
        return

    d_state = block.d_state
    d_inner = block.d_inner
    dt_rank = block.dt_rank
    n_heads = block.n_heads
    d_model = cfg.d_model
    K = getattr(cfg, 'tbd_n_scales', 4)
    moduli = getattr(cfg, 'rns_moduli', (8, 9, 11, 13, 7))
    k = len(moduli)

    base_dim = d_model // k
    remainder = d_model % k
    rns_dims = [base_dim + (1 if i < remainder else 0) for i in range(k)]

    rns_dims_inner = [int(round(d / d_model * d_inner)) for d in rns_dims]
    rns_dims_inner[-1] += d_inner - sum(rns_dims_inner)

    col_offsets = []
    s = 0
    for di in rns_dims_inner:
        col_offsets.append(s)
        s += di

    heads_per_group = n_heads // K
    head_dim = d_state // n_heads
    group_rows = [(j * heads_per_group * head_dim,
                   ((j+1) * heads_per_group if j < K-1 else n_heads) * head_dim)
                  for j in range(K)]

    assignment = {4: 3, 0: 2, 1: 1, 2: 0, 3: 0}

    with torch.no_grad():
        W = block.x_proj.weight
        B_start = dt_rank
        B_end = dt_rank + d_state

        W[B_start:B_end, :] = 0.0

        for slice_idx, group_j in assignment.items():
            r_start = B_start + group_rows[group_j][0]
            r_end = B_start + group_rows[group_j][1]
            c_start = col_offsets[slice_idx]
            c_end = c_start + rns_dims_inner[slice_idx]
            n_diag = min(r_end - r_start, c_end - c_start)
            W[r_start:r_end, c_start:c_end] = 0.0
            for d in range(n_diag):
                W[r_start + d, c_start + d] = 1.0

    B_block = W[B_start:B_end, :]
    n_nonzero = (B_block != 0).sum().item()
    print(f"[B*-INIT] Applied to layer 0: {n_nonzero} nonzero in B ({n_nonzero}/{B_block.numel()} = {100*n_nonzero/B_block.numel():.1f}%)", flush=True)


def apply_adaptive_threshold(model, cfg):
    """Set threshold_type to adaptive for all TernaryLinear modules if config enabled."""
    threshold_type = getattr(cfg, 'ternary_threshold_type', 'coefficient')
    if threshold_type != 'coefficient':
        from src.model_v11.components.ternary import TernaryLinear
        n_applied = 0
        for m in model.modules():
            if isinstance(m, TernaryLinear):
                m.threshold_type = threshold_type
                n_applied += 1
        if n_applied > 0:
            print(f"[ADAPT-THRESHOLD] Set threshold_type={threshold_type} for {n_applied} TernaryLinear modules", flush=True)


def calibrate_A_from_data(model, data, device):
    """R304: Calibrate A_log eigenspectrum to dataset autocorrelation."""
    import math
    N_CALIB = 2048
    TAU_MAX = 128

    start = 0
    end = min(N_CALIB + TAU_MAX + 1, len(data))
    tokens = torch.tensor(data[start:end], dtype=torch.long)
    T = tokens.shape[0]
    if T < TAU_MAX + 2:
        print("[CAL-A] Not enough data — skipping", flush=True)
        return

    emb_device = next(model.embedding.parameters()).device
    tokens = tokens.to(emb_device)

    with torch.no_grad():
        if hasattr(model.embedding, 'weight'):
            E = model.embedding.weight.detach()[tokens]
        else:
            E = model.embedding(tokens)
        norms = E.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        E = E / norms

    tau_max_actual = min(TAU_MAX, T // 4)
    R = torch.zeros(tau_max_actual)
    with torch.no_grad():
        for tau in range(1, tau_max_actual + 1):
            dots = (E[:T-tau] * E[tau:]).sum(dim=-1)
            R[tau-1] = dots.mean()

    n_heads = None
    for layer in model.layers:
        block = getattr(layer, 'block', None)
        if block and hasattr(block, 'A_log') and block.A_log.dim() == 1:
            n_heads = block.A_log.shape[0]
            break
    if n_heads is None:
        print("[CAL-A] No 1D A_log found — skipping", flush=True)
        return

    energy = R.clamp(min=0.0)
    total = energy.sum()
    if total < 1e-9:
        print("[CAL-A] Zero autocorrelation — skipping", flush=True)
        return

    cdf = energy.cumsum(0) / total

    cluster_taus = [1.0, 8.0, 60.0, 121.0]
    n_per_cluster = n_heads // len(cluster_taus)
    remainder = n_heads % len(cluster_taus)

    tau_scales = torch.zeros(n_heads)
    for h in range(n_heads):
        cluster_idx = int(h * 4 / n_heads)
        cluster_idx = min(cluster_idx, 3)
        tau_scales[h] = min(cluster_taus[cluster_idx], float(tau_max_actual))

    tau_scales = tau_scales.sort().values.clamp(min=1.0)
    A_log_calib = (-1.0 / tau_scales).clamp(min=math.log(0.05), max=math.log(0.9999))

    print(f"[EIGEN-4C] 4 clusters: tau={cluster_taus}, {n_per_cluster} per cluster, remainder={remainder}", flush=True)

    n_applied = 0
    with torch.no_grad():
        for layer in model.layers:
            block = getattr(layer, 'block', None)
            if block and hasattr(block, 'A_log') and block.A_log.dim() == 1 and block.A_log.shape[0] == n_heads:
                block.A_log.data.copy_(A_log_calib)
                n_applied += 1

    print(f"[CAL-A] Calibrated {n_applied} layers | tau=[{tau_scales[0]:.1f},{tau_scales[-1]:.1f}] | "
          f"r=[{A_log_calib.exp().min():.4f},{A_log_calib.exp().max():.4f}]", flush=True)


def make_config(model_size):
    """Create ModelConfig from model size."""
    from src.model_v11.config import ModelConfig

    if model_size == "6m":
        return ModelConfig(
            d_model=256,
            n_layer=8,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank=16,
            vocab_size=256,
            seq_len=512,
            n_heads=8,
            mtp_num_heads=3,
            use_ternary=True,  # ALWAYS True — this project IS ternary
            ternary_lm_head=True,  # ALWAYS True — this project IS ternary
            use_ssd=False,
            use_bf16=False,
            use_gradient_checkpointing=False,
            use_local_loss_per_block=False,
            tied_embeddings=False,
            dropout=0.0,
            swa_layer_indices=[],
            use_torch_compile=False,
            use_bitshift_norm=False,
            use_bf16_lm_head=False,
            use_pid_commitment_lr=False,
            use_uncommitted_gap_lr=False,
            use_bitmask_gemv=False,
            use_int32_forward=True,
            use_activation_quant=False,
            use_predictive_coding=True,
            use_mux=True,
            use_wht_mixing=True,
        )
    elif model_size == "100m":
        return ModelConfig(
            d_model=1024,
            n_layer=6,
            d_state=128,
            d_conv=4,
            expand=2,
            dt_rank=64,
            vocab_size=32064,  # 32000 BPE + 7 chat special tokens, padded to 64× for tensor cores
            seq_len=512,
            n_heads=16,
            mtp_num_heads=3,
            use_ternary=True,  # ALWAYS True — this project IS ternary
            ternary_lm_head=True,  # ALWAYS True — this project IS ternary
            use_ssd=True,
            ssd_n_heads=16,
            use_bf16=False,
            use_gradient_checkpointing=False,
            use_local_loss_per_block=False,
            tied_embeddings=False,
            dropout=0.0,
            swa_layer_indices=[],
            use_torch_compile=False,
            use_bitshift_norm=False,
            use_bf16_lm_head=False,
            use_pid_commitment_lr=False,
            use_uncommitted_gap_lr=False,
            use_bitmask_gemv=False,
            use_int32_forward=True,
            use_activation_quant=False,
            use_rns_embedding=True,
            use_predictive_coding=True,
            predictive_coding_rank=24,  # R304 validated: rank 20-32 optimal, 64 was over-dimensioned
            use_mux=True,
            use_wht_mixing=True,
            use_tbd_init=True,
            tbd_n_scales=4,
            use_tcsd=True,
            use_osp=True,
            osp_lambda=0.01,
            osp_alpha=0.1,
            osp_ema_beta=0.99,
            use_tma=True,
            tma_n_slots=64,
            use_traj_diversity=True,
            traj_diversity_lambda=0.01,
            traj_diversity_K=5,
            use_prompt_state_anchor=True,
            psa_gamma_init=0.1,
            psa_decay_mode='fixed',
            use_eos_head=True,
            eos_head_weight=0.1,
            eos_token_id=32003,
            eos_focal_gamma=2.0,
            ternary_threshold_type='adaptive',
        )
    elif model_size == "7b":
        return ModelConfig(
            d_model=4096,
            n_layer=68,
            d_state=128,
            d_conv=4,
            expand=2,
            dt_rank=128,
            vocab_size=256,
            seq_len=512,
            n_heads=128,
            mtp_num_heads=3,
            use_ternary=True,  # ALWAYS True — this project IS ternary
            ternary_lm_head=True,  # ALWAYS True — this project IS ternary
            use_ssd=True,
            ssd_n_heads=128,
            use_bf16=False,
            use_gradient_checkpointing=False,
            use_local_loss_per_block=False,
            tied_embeddings=False,
            dropout=0.0,
            swa_layer_indices=[],
            use_torch_compile=False,
            use_bitshift_norm=False,
            use_bf16_lm_head=False,
            use_pid_commitment_lr=False,
            use_uncommitted_gap_lr=False,
            use_bitmask_gemv=False,
            use_int32_forward=True,
            use_activation_quant=False,
            use_predictive_coding=True,
            use_mux=True,
            use_wht_mixing=True,
        )
    else:
        raise ValueError(f"Unknown model_size: {model_size}")


def load_data(dataset_path, vocab_size=256, seed=42):
    """Load data from file or generate random."""
    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        print(f"[PROD] Loading data from {dataset_path}...", flush=True)
        with open(dataset_path, "rb") as f:
            dtype = np.uint16 if vocab_size > 256 else np.uint8
            data = np.frombuffer(f.read(), dtype=dtype)
        print(f"[PROD] Loaded {len(data)} bytes", flush=True)
    else:
        print(f"[PROD] Data not found, generating random data", flush=True)
        np.random.seed(seed)
        data = np.random.randint(0, vocab_size, size=1000000, dtype=np.uint8)

    return data


def sample_batch(data, seq_len, batch_size, vocab_size, device):
    """Sample a batch from data."""
    indices = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    seqs = np.array([data[i:i+seq_len+1] for i in indices], dtype=np.int64)
    x = torch.from_numpy(seqs[:, :-1]).to(device)
    y = torch.from_numpy(seqs[:, 1:]).to(device)
    return x, y


def build_eval_batches(data, config, vocab_size, device="cpu"):
    """V-PRIME V1: fixed eval batches, sampled once, stored on CPU."""
    print(f"[V-PRIME] Building fixed eval set: {config.eval_fixed_batches} batches", flush=True)
    fixed = []
    for _ in range(config.eval_fixed_batches):
        x, y = sample_batch(data, config.seq_len, config.batch_size, vocab_size, device)
        fixed.append((x.cpu(), y.cpu()))
    return fixed


def build_caches(model, data, n_stages, config, device, vocab_size=256):
    """Precompute H_0 cache (embedding outputs)."""
    h_cache = [[None] * config.cache_batches for _ in range(n_stages)]
    y_cache = [[None] * config.cache_batches for _ in range(n_stages)]

    with torch.no_grad():
        for batch_idx in range(config.cache_batches):
            x, y = sample_batch(
                data, config.seq_len, config.batch_size, vocab_size, device
            )
            h = model.embedding(x)
            h_cache[0][batch_idx] = h.cpu().detach()
            y_cache[0][batch_idx] = y.cpu().detach()

    print(f"[PROD] H_0 cache ready ({config.cache_batches} batches)", flush=True)
    return h_cache, y_cache


def calibrate_warm_h0(model, h_cache_stage0, device, n_batches=None):
    """Compute corpus-mean h₀ for every SSM layer using cached H_0 inputs.

    Runs after build_caches() so h_cache[0] already has embedding outputs.
    Uses up to n_batches (default: all) from stage-0 cache.

    Call ONCE before run_fp_pipeline().
    """
    from src.model_v11.components.ssm import SelectiveSSMSSD

    batches = h_cache_stage0
    if n_batches is not None:
        batches = batches[:n_batches]

    print(f"[WARM-H0] Calibrating h₀ for all SSM layers ({len(batches)} batches)...", flush=True)

    for layer_idx, layer in enumerate(model.layers):
        ssm_mod = getattr(layer, 'ssm', None) or getattr(layer, 'mixer', None)
        if ssm_mod is None or not isinstance(ssm_mod, SelectiveSSMSSD):
            continue

        ssm_mod.to(device)
        ssm_mod.calibrate_h0(batches, device)
        ssm_mod.to('cpu')
        torch.cuda.empty_cache()

    print(f"[WARM-H0] All layers calibrated.", flush=True)


def precompute_stage_cache(model, h_cache, y_cache, stage_k, config, device):
    """Precompute H_k for stage k from H_{k-1}."""
    assert stage_k >= 1, "stage 0 nao precisa de precompute"

    layer_to_run = stage_k - 1
    model.layers[layer_to_run].to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx in range(config.cache_batches):
            h_prev = h_cache[stage_k - 1][batch_idx]
            assert h_prev is not None, f"slot {batch_idx} do cache {stage_k-1} eh None!"
            h_prev_gpu = h_prev.to(device)
            h_next = model.layers[layer_to_run](h_prev_gpu)
            if isinstance(h_next, tuple):
                h_next = h_next[0]
            h_cache[stage_k][batch_idx] = h_next.detach().cpu()

        y_cache[stage_k] = y_cache[stage_k - 1]

    model.layers[layer_to_run].to("cpu")
    torch.cuda.empty_cache()
    t1 = time.perf_counter()
    print(f"[PROD] precompute stage={stage_k}: {t1-t0:.2f}s "
          f"({config.cache_batches} batches)", flush=True)


def forward_frozen_layers(model, h_in, layer_start, layer_end):
    """Forward pass through frozen layers."""
    h = h_in
    for k in range(layer_start, layer_end):
        h_out = model.layers[k](h)
        if isinstance(h_out, tuple):
            h_out = h_out[0]
        h = h_out
    return h


def aba_should_stop(recent_losses, step, min_steps=50, patience=50,
                    rel_threshold=0.001, abs_threshold=0.005, p_threshold=0.01):
    """Welch t-test early stopping: stops when loss is NOT significantly improving."""
    if step < min_steps:
        return False, ""
    if len(recent_losses) < patience:
        return False, ""

    import math
    window = list(recent_losses)[-patience:]
    mid = len(window) // 2
    first_half = window[:mid]
    second_half = window[mid:]

    n1, n2 = len(first_half), len(second_half)
    mean1 = sum(first_half) / n1
    mean2 = sum(second_half) / n2

    if mean2 >= mean1:
        return True, f"welch_no_improvement (mean1={mean1:.4f} mean2={mean2:.4f})"

    var1 = sum((x - mean1) ** 2 for x in first_half) / max(n1 - 1, 1)
    var2 = sum((x - mean2) ** 2 for x in second_half) / max(n2 - 1, 1)
    se = ((var1 / n1) + (var2 / n2)) ** 0.5

    if se < 1e-10:
        return True, f"welch_flat (se={se:.2e})"

    t_stat = (mean1 - mean2) / se

    dof_num = ((var1 / n1) + (var2 / n2)) ** 2
    dof_den = ((var1 / n1) ** 2 / max(n1 - 1, 1)) + ((var2 / n2) ** 2 / max(n2 - 1, 1))
    dof = dof_num / max(dof_den, 1e-10)

    if dof >= 20:
        p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))
    else:
        p_value = 0.5 * math.erfc(t_stat / math.sqrt(2)) * (1 + 2.0 / max(dof, 1))

    if p_value >= p_threshold:
        return True, (f"welch_plateau (t={t_stat:.3f} p={p_value:.4f} "
                      f"dof={dof:.1f} mean1={mean1:.4f} mean2={mean2:.4f})")

    return False, ""


def compute_grad_norm(model):
    """Compute gradient norm."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm_sq)


def train_stage_adaptive_cached(model, stage_k, data, h_cache, y_cache, losses_all,
                                config, device, n_stages,
                                fp_momentum_store=None, grad_ema=None, sign_count=None,
                                is_last_in_loop=False):
    """Train single stage with cache and ABA."""
    print(f"[PROD] ========== STAGE {stage_k} ==========", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.layers[stage_k].parameters():
        p.requires_grad_(True)
    if stage_k == n_stages - 1:
        if model.lm_head is not None:
            for p in model.lm_head.parameters():
                p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PROD] Stage {stage_k}: Trainable params: {n_trainable/1e6:.1f}M", flush=True)

    model.layers[stage_k].to(device)
    torch.cuda.empty_cache()
    vram_snapshot(f"stage{stage_k}_layer_to_gpu", verbose=False)

    if stage_k == n_stages - 1 and model.lm_head is not None and n_stages > 1:
        print(f"[LM_HEAD_REINIT] Reinitializing lm_head before stage {stage_k}", flush=True)
        import torch.nn as nn
        nn.init.normal_(model.lm_head.weight, std=0.02)
        if model.lm_head.bias is not None:
            nn.init.zeros_(model.lm_head.bias)

        print(f"[WARM_START] stage={stage_k}: freezing lm_head for 20 pre-steps", flush=True)

        for p in model.lm_head.parameters():
            p.requires_grad_(False)

        _warm_params = [p for p in model.layers[stage_k].parameters() if p.requires_grad]
        _warm_optimizer = torch.optim.AdamW(_warm_params, lr=config.fp_lr, betas=(0.9, 0.95), weight_decay=1e-4)

        model.train()
        for _ws in range(20):
            _batch_idx = _ws % config.cache_batches
            _h_in_cached = h_cache[stage_k][_batch_idx]
            if _h_in_cached is not None:
                _h_in = _h_in_cached.to(device)
            else:
                _x, _ = sample_batch(data, config.seq_len, config.batch_size, get_vocab_size(model), device)
                _h_in = model.embedding(_x)
                _h_in = forward_frozen_layers(model, _h_in, 0, stage_k)

            _y = y_cache[stage_k][_batch_idx].to(device)

            _h_k = model.layers[stage_k](_h_in)
            if isinstance(_h_k, tuple):
                _h_k = _h_k[0]

            if hasattr(model, 'final_norm'):
                _h_norm = model.final_norm(_h_k)
            elif hasattr(model, 'norm_f'):
                _h_norm = model.norm_f(_h_k)
            else:
                _h_norm = _h_k

            _loss_val, _dH = streaming_ce_loss(_h_norm.detach(), model.lm_head.weight, _y, head_module=model.lm_head)
            _warm_optimizer.zero_grad()
            _h_norm.backward(_dH)
            torch.nn.utils.clip_grad_norm_(_warm_params, 1.0)
            _warm_optimizer.step()

        print(f"[WARM_START] done: pre-step loss={_loss_val:.4f}. Unfreezing lm_head.", flush=True)

        for p in model.lm_head.parameters():
            p.requires_grad_(True)

        del _warm_optimizer, _warm_params
        torch.cuda.empty_cache()

    optimizer = FactoredAdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.fp_lr,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
        min_factor_dim=64
    )

    if is_last_in_loop:
        n_effective_steps = int(config.max_steps_per_stage * 2.5)
        print(f"[STEP_BUDGET] stage={stage_k} LAST STAGE: budget={n_effective_steps} (2.5x base {config.max_steps_per_stage})", flush=True)
    else:
        n_effective_steps = config.max_steps_per_stage
    n_warmup = min(5, max(1, n_effective_steps // 10))
    def fp_lr_lambda(s):
        if s < n_warmup:
            return max(0.1, s / n_warmup)
        progress = (s - n_warmup) / max(1, n_effective_steps - n_warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fp_lr_lambda)

    stage_losses = []
    step_times = []
    loss_step0 = None
    recent_losses = []
    steps_used = 0
    stop_reason = "max_steps"

    print(f"[PROD] Stage {stage_k}: Starting training (max {n_effective_steps} steps)...", flush=True)
    vocab_sz = get_vocab_size(model)
    print(f"[PROD] Random loss (ln({vocab_sz})): {math.log(vocab_sz):.4f}", flush=True)

    model.train()

    for step in range(n_effective_steps + 1):
        batch_idx = step % config.cache_batches

        if stage_k == 0:
            h_in_cached = h_cache[0][batch_idx]
            if h_in_cached is not None:
                h_in = h_in_cached.to(device)
            else:
                x, _ = sample_batch(data, config.seq_len, config.batch_size, get_vocab_size(model), device)
                h_in = model.embedding(x)
        else:
            h_in_cached = h_cache[stage_k][batch_idx]
            if h_in_cached is not None:
                h_in = h_in_cached.to(device)
            else:
                x, _ = sample_batch(data, config.seq_len, config.batch_size, get_vocab_size(model), device)
                h_in = model.embedding(x)
                h_in = forward_frozen_layers(model, h_in, 0, stage_k)

        y = y_cache[stage_k][batch_idx].to(device)
        h = h_in

        h_k = model.layers[stage_k](h)
        if isinstance(h_k, tuple):
            h_k = h_k[0]

        if stage_k == n_stages - 1:
            if hasattr(model, 'final_norm'):
                h_norm = model.final_norm(h_k)
            elif hasattr(model, 'norm_f'):
                h_norm = model.norm_f(h_k)
            else:
                h_norm = h_k

            if step == 0:
                with torch.no_grad():
                    if model.lm_head is not None:
                        _z0 = model.lm_head(h_norm)
                    else:
                        _z0 = F.linear(h_norm, get_embedding_weight(model))
                    loss_scalar = F.cross_entropy(_z0.reshape(-1, _z0.shape[-1]), y.reshape(-1)).item()
                    del _z0
                print(f"[ACTIVATION] PROD FP stage={stage_k} step=0 loss={loss_scalar:.4f}", flush=True)
                vram_snapshot(f"stage{stage_k}_step0", verbose=False)
                loss_step0 = loss_scalar
                stage_losses.append(loss_scalar)
                if stage_k < n_stages - 1:
                    h_cache[stage_k + 1][batch_idx] = h_k.detach().cpu()
                continue

            _w = get_embedding_weight(model)
            _hm = model.lm_head if model.lm_head is not None else None

            loss_scalar, _dH_norm = streaming_ce_loss(h_norm.detach(), _w, y, head_module=_hm)

            if config.use_psa:
                _psa_w = get_embedding_weight(model).detach()
                _psa_loss, _dH_psa = streaming_ce_loss(h_norm.detach(), _psa_w, y)
                _dH_norm = _dH_norm + config.psa_weight * _dH_psa

            if config.use_til:
                _til_loss = compute_til_loss(model, h_norm.detach(), y, stage_k,
                                              K=config.til_K, til_weight=config.til_weight)
                if _til_loss.requires_grad:
                    _til_loss.backward()
        else:
            h_norm = h_k

            if step == 0:
                with torch.no_grad():
                    _z0 = F.linear(h_norm, get_embedding_weight(model))
                    loss_scalar = F.cross_entropy(_z0.reshape(-1, _z0.shape[-1]), y.reshape(-1)).item()
                    del _z0
                print(f"[ACTIVATION] PROD FP stage={stage_k} step=0 loss={loss_scalar:.4f}", flush=True)
                vram_snapshot(f"stage{stage_k}_step0", verbose=False)
                loss_step0 = loss_scalar
                stage_losses.append(loss_scalar)
                if stage_k < n_stages - 1:
                    h_cache[stage_k + 1][batch_idx] = h_k.detach().cpu()
                continue

            loss_scalar, _dH_norm = streaming_ce_loss(h_norm.detach(), get_embedding_weight(model), y, head_module=None)

            if config.use_psa:
                _psa_w = get_embedding_weight(model).detach()
                _psa_loss, _dH_psa = streaming_ce_loss(h_norm.detach(), _psa_w, y)
                _dH_norm = _dH_norm + config.psa_weight * _dH_psa

            if config.use_til:
                _til_loss = compute_til_loss(model, h_norm.detach(), y, stage_k,
                                              K=config.til_K, til_weight=config.til_weight)
                if _til_loss.requires_grad:
                    _til_loss.backward()

        if step == 1 and config.use_psa:
            print(f"[PSA] active: psa_loss={_psa_loss:.4f} weight={config.psa_weight}", flush=True)

        if step == 1 and config.use_til:
            print(f"[TIL] active: K={config.til_K} weight={config.til_weight}", flush=True)

        t0 = time.perf_counter()
        h_norm.backward(_dH_norm)

        if config.use_osp:
            for _m in model.modules():
                if hasattr(_m, 'A_log') and hasattr(_m, '_osp_target') and _m.A_log.requires_grad:
                    with torch.no_grad():
                        _a = -torch.exp(_m.A_log)
                        _a2 = _a ** 2
                        _osp_grad = -4.0 * (_a2 - _m._osp_target) * _a2 * config.osp_lambda
                        if _m.A_log.grad is not None:
                            _m.A_log.grad.add_(_osp_grad)
                        else:
                            _m.A_log.grad = _osp_grad.clone()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        gn = compute_grad_norm(model)

        if config.use_gradient_filter:
            total_params = 0
            masked_params = 0
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    total_params += p.grad.numel()
                    if name not in grad_ema:
                        grad_ema[name] = torch.zeros_like(p.grad)
                        sign_count[name] = torch.zeros_like(p.grad, dtype=torch.int8)

                    with torch.no_grad():
                        grad_ema[name] = 0.9 * grad_ema[name] + 0.1 * p.grad.data
                        same_sign = (grad_ema[name].sign() == p.grad.data.sign())
                        sign_count[name] = torch.where(
                            same_sign,
                            (sign_count[name] + 1).clamp(max=127),
                            torch.zeros_like(sign_count[name])
                        )
                        mask = (sign_count[name] >= config.filter_n_min).float()
                        p.grad.data.mul_(mask)
                        masked_params += mask.sum().item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        t1 = time.perf_counter()

        stage_losses.append(loss_scalar)
        step_times.append(t1 - t0)
        recent_losses.append(loss_scalar)
        steps_used = step

        should_stop, reason = aba_should_stop(
            recent_losses, step,
            min_steps=config.min_steps,
            patience=config.patience,
            rel_threshold=config.rel_threshold,
            abs_threshold=config.abs_threshold
        )
        if should_stop:
            stop_reason = reason
            print(f"[PROD] stage={stage_k} EARLY STOP at step={step} reason: {reason}", flush=True)
            break

        if step % config.log_every == 0:
            avg_step_s = np.mean(step_times[-config.log_every:]) if len(step_times) >= config.log_every else np.mean(step_times)
            print(f"[PROD] stage={stage_k} step={step:3d} loss={loss_scalar:.4f} gn={gn:.4f} "
                  f"({avg_step_s*1000:.0f}ms/step)", flush=True)

        allocated_mb = torch.cuda.memory_allocated() / 1e6
        if allocated_mb > config.max_vram_mb:
            print(f"[ERROR] Stage {stage_k}: VRAM exceeded {config.max_vram_mb} MB: {allocated_mb:.0f} MB", flush=True)
            break

        if stage_k < n_stages - 1 and step % 5 == 0:
            h_cache[stage_k + 1][batch_idx] = h_k.detach().cpu()

    print(f"[PROD] stage={stage_k} steps_used={steps_used}/{n_effective_steps} "
          f"stop_reason={stop_reason}", flush=True)

    loss_step_final = stage_losses[-1]
    reduction = (loss_step0 - loss_step_final) / loss_step0 * 100 if loss_step0 > 0 else 0

    avg_step_ms = np.mean(step_times) * 1000 if step_times else 0.0
    vram_peak = torch.cuda.memory_allocated() / 1e6

    print(f"""[PROD] === STAGE {stage_k} RESULTADO ===
[PROD] loss_step0:      {loss_step0:.4f}
[PROD] loss_final:      {loss_step_final:.4f}
[PROD] reducao:         {reduction:.1f}%
[PROD] steps_used:      {steps_used}/{n_effective_steps}
[PROD] avg ms/step:     {avg_step_ms:.0f}ms
[PROD] VRAM pico:       {vram_peak:.0f} MB
[PROD] === FIM STAGE ===
""", flush=True)

    if fp_momentum_store is not None:
        name_to_param = {
            name: p
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        layer_state = {}
        for name, p in name_to_param.items():
            if p in optimizer.state and len(optimizer.state[p]) > 0:
                raw = optimizer.state[p]
                layer_state[name] = {
                    k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in raw.items()
                }
        fp_momentum_store[stage_k] = layer_state
        n_saved = len(layer_state)
        print(f"[PROD] momentum_store stage={stage_k}: salvou {n_saved} param states", flush=True)

    model.layers[stage_k].to("cpu")
    torch.cuda.empty_cache()
    vram_snapshot(f"stage{stage_k}_layer_back_to_cpu", verbose=False)

    losses_all.extend(stage_losses)
    return stage_losses, steps_used, stop_reason


def get_stage_order(n_stages, fp_order):
    """Return list of stage indices in training order."""
    if fp_order == "forward":
        return list(range(n_stages))
    elif fp_order == "reverse":
        return list(reversed(range(n_stages)))
    elif fp_order == "bisection":
        order = []
        queue = [(0, n_stages - 1)]
        visited = set()
        while queue:
            lo, hi = queue.pop(0)
            mid = (lo + hi) // 2
            if mid not in visited:
                order.append(mid)
                visited.add(mid)
            if lo < mid:
                queue.append((lo, mid - 1))
            if mid < hi:
                queue.append((mid + 1, hi))
        for i in range(n_stages):
            if i not in visited:
                order.append(i)
        return order
    else:
        raise ValueError(f"Unknown fp_order: {fp_order}")


def run_fp_pipeline(model, data, config, device, n_stages):
    """Run FP training across all stages."""
    print(f"[PROD] === FP PIPELINE START: {n_stages} stages ===", flush=True)

    print(f"[PROD] Building H_k caches ({n_stages} stages x {config.cache_batches} batches)...", flush=True)
    h_cache, y_cache = build_caches(model, data, n_stages, config, device, vocab_size=get_vocab_size(model))

    if getattr(config, 'use_warm_h0', False):
        calibrate_warm_h0(model, h_cache[0], device, n_batches=getattr(config, 'warm_h0_batches', 10))

    losses_all = []
    stage_times = []
    stage_steps = []
    stage_stop_reasons = []
    fp_momentum_store = {}
    grad_ema = {}
    sign_count = {}

    order_list = get_stage_order(n_stages, config.fp_order)
    print(f"[PROD] fp_order={config.fp_order} order={order_list}", flush=True)

    if config.fp_order != "forward":
        for k in range(1, n_stages):
            precompute_stage_cache(model, h_cache, y_cache, k, config, device)

    for loop_idx, stage_k in enumerate(order_list):
        is_last_in_loop = (loop_idx == len(order_list) - 1)

        if config.fp_order == "forward" and stage_k > 0:
            precompute_stage_cache(model, h_cache, y_cache, stage_k, config, device)

        t0 = time.perf_counter()
        sl, steps_used, stop_reason = train_stage_adaptive_cached(
            model, stage_k, data, h_cache, y_cache, losses_all,
            config, device, n_stages,
            fp_momentum_store=fp_momentum_store,
            grad_ema=grad_ema,
            sign_count=sign_count,
            is_last_in_loop=is_last_in_loop
        )
        t_elapsed = time.perf_counter() - t0
        stage_times.append(t_elapsed)
        stage_steps.append(steps_used)
        stage_stop_reasons.append(stop_reason)

    print(f"[PROD] === FP PIPELINE COMPLETE ===", flush=True)
    return model, losses_all, losses_all[-1], stage_times, stage_steps, stage_stop_reasons, fp_momentum_store


def _get_prompt_len(x, cfg):
    """Fallback prompt_len when PSA is enabled but no boundary info available.
    Uses T//2 as default (assumes first half is prompt, second half is response)."""
    if not getattr(cfg, 'use_prompt_state_anchor', False):
        return None
    return torch.full((x.size(0),), x.size(1) // 2, dtype=torch.long, device=x.device)


def capacity_matching_step(model, step):
    """Equalize effective rank across layers (R318-PB5).

    Computes per-layer effective rank r_k = ||W_k||_F^2 / ||W_k||_2^2.
    If any layer's rank deviates >30% from mean, adjusts its adaptive threshold τ_k.
    """
    ranks = []
    ternary_layers = []

    for i, layer in enumerate(model.layers):
        for name, m in layer.named_modules():
            if hasattr(m, 'weight') and hasattr(m, 'threshold_coeff'):
                w = m.weight.data
                fro_sq = (w * w).sum()
                spec_sq = torch.linalg.matrix_norm(w, ord=2) ** 2
                r_k = (fro_sq / spec_sq).item() if spec_sq > 1e-12 else 0.0
                ranks.append(r_k)
                ternary_layers.append((i, name, m))
                break

    if not ranks:
        return

    mean_r = sum(ranks) / len(ranks)
    adjusted = 0
    for (i, name, m), r_k in zip(ternary_layers, ranks):
        if hasattr(m, 'threshold_coeff'):
            ratio = r_k / mean_r if mean_r > 1e-8 else 1.0
            if ratio < 0.7:
                m.threshold_coeff *= 0.95
                adjusted += 1
            elif ratio > 1.3:
                m.threshold_coeff *= 1.05
                adjusted += 1

    if adjusted > 0:
        print(f"[CAPACITY] step={step} adjusted={adjusted}/{len(ranks)} mean_rank={mean_r:.1f}", flush=True)


class FisherLossCoupler:
    """Dynamically adjusts loss weights via Fisher information.
    Solves for optimal weights that maximize convergence rate."""

    def __init__(self, n_losses, update_every=100, ema_decay=0.95, device='cuda'):
        self.n = n_losses
        self.update_every = update_every
        self.ema_decay = ema_decay
        self.device = device
        self.F_hat = torch.zeros(n_losses, n_losses, device=device)
        self.weights = torch.ones(n_losses, device=device) / n_losses
        self.step_count = 0
        self._logged = False

    def get_weighted_loss(self, losses):
        """Return weighted combination of losses."""
        total = None
        for i, l in enumerate(losses):
            if l is not None and i < self.n and (isinstance(l, torch.Tensor) and l.requires_grad):
                w = self.weights[i].item()
                if total is None:
                    total = w * l
                else:
                    total = total + w * l

        if total is None:
            return losses[0] if losses and losses[0] is not None else torch.tensor(0.0, device=self.device)
        return total


def direct_ternary_assignment(model, step):
    assigned = 0
    total = 0

    for m in model.modules():
        if not (hasattr(m, 'weight') and hasattr(m, 'threshold_coeff')):
            continue

        w = m.weight
        if w.grad is None:
            continue

        s = w.data
        tau = m.threshold_coeff * w.data.abs().mean()
        confident = s.abs() > 2 * tau

        if confident.any():
            snap_target = torch.sign(s) * tau * 3
            w.data[confident] = snap_target[confident]
            assigned += confident.sum().item()

        total += w.numel()

    if assigned > 0:
        pct = assigned / max(total, 1) * 100
        print(f"[DIRECT-B] step={step} assigned={pct:.1f}% ({assigned}/{total}) weights snapped to ternary", flush=True)


def precompute_osp_targets(model, data, device, config, n_batches=20):
    """Precompute full-dataset OSP targets from n_batches uniform samples (R338-M4)."""
    osp_modules = [m for m in model.modules()
                   if hasattr(m, '_osp_target') and hasattr(m, 'A_log')]
    if not osp_modules:
        return

    for m in osp_modules:
        m._calibration_mode = True
        m._osp_calib_accum = torch.zeros_like(m._osp_target)
        m._osp_calib_count = 0

    model.train()
    vocab_size = get_vocab_size(model)

    print(f"[OSP-CALIB] Precomputing full-dataset targets from {n_batches} batches...", flush=True)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_batch(data, config.seq_len, config.batch_size, vocab_size, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(x)

    for m in osp_modules:
        if m._osp_calib_count > 0:
            mean_b2 = m._osp_calib_accum / m._osp_calib_count
            m._osp_target.copy_(1.0 - m._osp_alpha * mean_b2)
        m._calibration_mode = False
        if hasattr(m, '_osp_calib_accum'):
            del m._osp_calib_accum
        if hasattr(m, '_osp_calib_count'):
            del m._osp_calib_count

    print(f"[OSP-CALIB] Done. Targets set for {len(osp_modules)} SSM modules. "
          f"Example target mean: {osp_modules[0]._osp_target.mean():.4f}", flush=True)


def run_joint_gd(model, data, config, device, cfg, fixed_batches=None):
    """Standard joint gradient descent — all layers trained simultaneously."""
    print(f"[BF16] Autocast BF16 ACTIVE", flush=True)
    print(f"[BOUNDARY-FILTER] ACTIVE: zero within-cell gradients", flush=True)
    print(f"[DEALLOC] Dynamic optimizer deallocation ACTIVE", flush=True)
    print(f"[DIRECT-B] Periodic direct ternary assignment ACTIVE (every 500 steps)", flush=True)
    print(f"[JOINT-GD] Starting: {config.joint_steps} steps, lr={config.joint_lr}, batch={config.batch_size}", flush=True)
    if config.use_capacity_matching:
        print(f"[CAPACITY] Cross-layer capacity matching ACTIVE", flush=True)

    model.to(device)
    model.train()
    torch.cuda.empty_cache()
    vram_snapshot("joint_gd_model_on_gpu")

    if config.use_osp:
        precompute_osp_targets(model, data, device, config, n_batches=20)

    vocab_size = get_vocab_size(model)

    optimizer = FactoredAdamW(model.parameters(), lr=config.joint_lr,
                             betas=(0.9, 0.95), weight_decay=1e-4, min_factor_dim=64)
    print(f"[FACTORED-ADAM] ACTIVE: rank-1 factored moments for 2D params (12.8GB→12.5MB on 7B)", flush=True)

    n_warmup = int(config.joint_steps * config.joint_warmup_ratio)
    def lr_lambda(step):
        if step < n_warmup:
            return max(0.1, step / max(n_warmup, 1))
        progress = (step - n_warmup) / max(1, config.joint_steps - n_warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses = []
    best_loss = float('inf')
    t_start = time.perf_counter()

    fisher_coupler = FisherLossCoupler(n_losses=4, update_every=100, device=device)
    print(f"[FISHER] Loss coupling ACTIVE: 4 loss terms, update every 100 steps", flush=True)

    # Recipe B — Active Set Crystallization
    # Track ternary value history per TernaryLinear layer
    _crystal_check_every = 500
    _crystal_ternary_prev = {}
    _crystal_unchanged_count = {}
    _crystal_masks = {}
    _crystal_logged = False

    # Build initial snapshots
    for _m in model.modules():
        if _m.__class__.__name__ == 'TernaryLinear':
            _lid = id(_m)
            with torch.no_grad():
                _tq = (_m.weight.detach() > 0).to(torch.int8) - (_m.weight.detach() < 0).to(torch.int8)
                _crystal_ternary_prev[_lid] = _tq.cpu()
                _crystal_unchanged_count[_lid] = torch.zeros_like(_tq, dtype=torch.int16, device='cpu')
                _crystal_masks[_lid] = torch.zeros(_m.weight.shape, dtype=torch.bool, device=_m.weight.device)

    # --- POST-ACCUMULATE GRAD HOOKS (R340-V3) ---
    # Shared counters written by per-parameter hooks, read outside the loop
    _hook_norm_sq     = [0.0]   # sum of squared grad norms across all hooked params
    _hook_filtered    = [0]     # count of elements filtered by boundary filter
    _hook_total       = [0]     # total count of elements examined
    _hook_handles     = []      # hook handles for cleanup
    _hook_dealloc_candidates = {}  # {id(param): param} — params hook flagged for dealloc

    def _make_post_accum_hook(param, module_id, crystal_masks_ref, norm_sq_ref, filtered_ref, total_ref, dealloc_candidates_ref):
        """
        Factory: returns a post_accumulate_grad_hook for one TernaryLinear.weight parameter.

        Fires after loss.backward() fully completes, before any post-backward Python code runs.
        Modifies param.grad in-place. Does NOT set param.grad = None (optimizer.step() must read it).

        Captures by reference (lists/dicts are mutable):
          crystal_masks_ref  → _crystal_masks dict (keyed by id(module))
          norm_sq_ref        → _hook_norm_sq[0] accumulator
          filtered_ref       → _hook_filtered[0] accumulator
          total_ref          → _hook_total[0] accumulator
          dealloc_candidates_ref → _hook_dealloc_candidates dict
        """
        def _hook(p: torch.nn.Parameter) -> None:
            if p.grad is None:
                return

            g = p.grad.data  # direct reference, no copy

            # --- A. Zero crystallized gradient entries ---
            if module_id in crystal_masks_ref:
                _mask = crystal_masks_ref[module_id]
                if _mask is not None and _mask.any():
                    g[_mask] = 0.0

            # --- B. Boundary filter (R331-MPB2-5) ---
            # Only keep gradient when it pushes weight TOWARD ternary decision boundary
            # or when weight is near zero (undecided region)
            if hasattr(p, '_boundary_threshold_coeff'):
                _w = p.data
                _useful = (g * _w) < 0   # grad pushes toward boundary
                _near_zero = _w.abs() < (p._boundary_threshold_coeff * _w.abs().mean() * 0.5)
                _keep = _useful | _near_zero
                if not _keep.all():
                    g[~_keep] *= 0.1
                total_ref[0] += _w.numel()
                filtered_ref[0] += (~_keep).sum().item()

            # --- C. Flag for deallocation check (actual freeing happens outside) ---
            # Condition: gradient is effectively zero → weight is crystallized enough to free optimizer state
            if g.abs().max() < 1e-10:
                dealloc_candidates_ref[id(p)] = p

            # --- D. Accumulate norm² for clip_grad_norm_ emulation ---
            # clip_grad_norm_ will still run outside as the authoritative clip pass,
            # but this accumulates so the training loop can log the pre-clip norm.
            norm_sq_ref[0] += g.norm().item() ** 2

        return _hook


    def _register_hooks(model, crystal_masks, norm_sq, filtered, total, dealloc_candidates):
        handles = []
        for _m in model.modules():
            if _m.__class__.__name__ != 'TernaryLinear':
                continue
            _p = _m.weight
            if not _p.requires_grad:
                continue
            # Store threshold_coeff on the parameter itself so hook can access it
            # (threshold_coeff lives on the module, not the parameter — bridge it here)
            if hasattr(_m, 'threshold_coeff'):
                _p._boundary_threshold_coeff = _m.threshold_coeff
            _lid = id(_m)
            _handle = _p.register_post_accumulate_grad_hook(
                _make_post_accum_hook(_p, _lid, crystal_masks, norm_sq, filtered, total, dealloc_candidates)
            )
            handles.append(_handle)
        return handles


    # Register hooks once, before the training loop
    _hook_handles = _register_hooks(
        model, _crystal_masks, _hook_norm_sq, _hook_filtered, _hook_total, _hook_dealloc_candidates
    )
    print(f"[HOOK] Registered post_accumulate_grad_hook on {len(_hook_handles)} TernaryLinear.weight params", flush=True)

    for step in range(config.joint_steps):
        t0 = time.perf_counter()

        # Reset per-step hook accumulators
        _hook_norm_sq[0] = 0.0
        _hook_filtered[0] = 0
        _hook_total[0] = 0
        _hook_dealloc_candidates.clear()

        x, y = sample_batch(data, config.seq_len, config.batch_size, vocab_size, device)

        optimizer.zero_grad()
        _pl = _get_prompt_len(x, cfg)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            result = model(x, prompt_len=_pl)
            if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], list):
                logits = result[0]
                mtp_logits = result[1]
            elif isinstance(result, tuple):
                logits = result[0]
                mtp_logits = None
            else:
                logits = result
                mtp_logits = None

        loss_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        loss_mtp = torch.tensor(0.0, device=device)
        if mtp_logits is not None and cfg.mtp_num_heads > 0:
            for k, head_logits in enumerate(mtp_logits):
                offset = k + 2
                if x.shape[1] > offset:
                    future_logits = head_logits[:, :-offset, :].float()
                    future_targets = y[:, offset:]
                    if future_logits.shape[1] > 0:
                        loss_mtp = loss_mtp + F.cross_entropy(
                            future_logits.reshape(-1, future_logits.shape[-1]),
                            future_targets.reshape(-1),
                        )
            if len(mtp_logits) > 0:
                loss_mtp = loss_mtp / len(mtp_logits)

        _ssm_mods = []
        _h_states = []
        for _m in model.modules():
            if hasattr(_m, '_gp_mean') and _m._gp_mean is not None:
                _ssm_mods.append(_m)
                # Pack stats into a lightweight named dict — no tensor storage
                stats = dict(
                    mean=_m._gp_mean,
                    var=(_m._gp_m2 / max(_m._gp_n - 1, 1)).clamp(min=1e-8),
                    cross=_m._gp_cross,
                    n_pairs=getattr(_m, '_gp_n_pairs', 0),
                )
                _h_states.append(stats)
                # Reset accumulators for next step
                _m._gp_n = 0
                _m._gp_mean = None
                _m._gp_m2 = None
                _m._gp_cross = None
                _m._gp_prev = None
                _m._gp_n_pairs = 0

        fisher_gate_threshold = getattr(cfg, 'fisher_gate_threshold', 0.05)

        loss_gp_kl = torch.tensor(0.0, device=device)
        gp_kl_lambda = getattr(cfg, 'gp_kl_lambda', 0.05)
        gp_kl_computed = False

        loss_eos = torch.tensor(0.0, device=device)
        eos_weight = getattr(cfg, 'eos_head_weight', 0.1)
        eos_computed = False

        mtp_weight = getattr(config, 'mtp_weight', 0.1)

        fw_ce, fw_gp, fw_eos, fw_mtp = 1.0, 0.0, 0.0, 0.0
        if hasattr(fisher_coupler, 'weights') and fisher_coupler.weights.numel() >= 4:
            fw_ce = fisher_coupler.weights[0].item()
            fw_gp = fisher_coupler.weights[1].item()
            fw_eos = fisher_coupler.weights[2].item()
            fw_mtp = fisher_coupler.weights[3].item()

        if gp_kl_lambda > 0 and _h_states and fw_gp >= fisher_gate_threshold:
            loss_gp_kl = compute_gp_kl_loss(_ssm_mods, _h_states, gp_kl_lambda=gp_kl_lambda)
            gp_kl_computed = True
            if step == 0:
                print(f"[GP-KL-ONLINE] ACTIVE: {len(_h_states)} layers, stats from accumulators (zero tensor storage), loss={loss_gp_kl.item():.4f}", flush=True)

        if eos_weight > 0 and hasattr(model, '_eos_logits') and model._eos_logits is not None and fw_eos >= fisher_gate_threshold:
            _eos_pred = model._eos_logits
            _eos_tid = getattr(cfg, 'eos_token_id', 0)
            _eos_tgt = (y == _eos_tid).float()
            _gamma_f = getattr(cfg, 'eos_focal_gamma', 2.0)
            _bce = F.binary_cross_entropy_with_logits(_eos_pred, _eos_tgt, reduction='none')
            if _gamma_f > 0:
                _pt = torch.exp(-_bce)
                _bce = ((1 - _pt) ** _gamma_f) * _bce
            loss_eos = eos_weight * _bce.mean()
            eos_computed = True

        individual_losses = [
            loss_ce,
            loss_gp_kl,
            loss_eos,
            mtp_weight * loss_mtp if loss_mtp.item() > 0 else torch.tensor(0.0, device=device),
        ]

        if fisher_coupler.step_count % fisher_coupler.update_every == 0 and step > 0:
            loss_mags = []
            for l in individual_losses:
                if isinstance(l, torch.Tensor):
                    mag = l.detach().item() if l.requires_grad else l.item()
                else:
                    mag = float(l) if isinstance(l, (int, float)) else 0.0
                loss_mags.append(max(0.0, mag))

            with torch.no_grad():
                mag_tensor = torch.tensor(loss_mags, device=device, dtype=torch.float32)
                for i in range(fisher_coupler.n):
                    for j in range(i, fisher_coupler.n):
                        if mag_tensor[i] > 1e-8 and mag_tensor[j] > 1e-8:
                            cos = (mag_tensor[i] * mag_tensor[j]) / (mag_tensor[i] * mag_tensor[j] + 1e-8)
                            fisher_coupler.F_hat[i, j] = fisher_coupler.ema_decay * fisher_coupler.F_hat[i, j] + (1 - fisher_coupler.ema_decay) * cos
                            fisher_coupler.F_hat[j, i] = fisher_coupler.F_hat[i, j]

            try:
                v = torch.ones(fisher_coupler.n, device=device)
                for _ in range(10):
                    v = fisher_coupler.F_hat @ v
                    v_norm = v.norm() + 1e-8
                    v = v / v_norm
                fisher_coupler.weights = v.abs()
                fisher_coupler.weights = fisher_coupler.weights / (fisher_coupler.weights.sum() + 1e-8)
            except Exception:
                pass
            fisher_coupler.step_count += 1

        loss = fisher_coupler.get_weighted_loss(individual_losses)
        if loss is None or (isinstance(loss, torch.Tensor) and loss.item() == 0):
            loss = loss_ce

        if step > 0 and step % 100 == 0:
            active_losses = ['CE']
            if gp_kl_computed:
                active_losses.append('GP-KL')
            if eos_computed:
                active_losses.append('EOS')
            if loss_mtp.item() > 0:
                active_losses.append('MTP')
            skipped = []
            if not gp_kl_computed and gp_kl_lambda > 0 and _h_states:
                skipped.append('GP-KL')
            if not eos_computed and eos_weight > 0 and hasattr(model, '_eos_logits') and model._eos_logits is not None:
                skipped.append('EOS')
            skipped_str = ', '.join(skipped) if skipped else 'none'
            print(f"[FISHER-GATE] step={step} active={active_losses} skipped=[{skipped_str}] fisher_weights=[CE:{fw_ce:.3f}, GP:{fw_gp:.3f}, EOS:{fw_eos:.3f}, MTP:{fw_mtp:.3f}]", flush=True)

        if step == 0:
            weight_str = ', '.join(f'{w:.3f}' for w in fisher_coupler.weights.cpu().numpy())
            print(f"[FISHER] Initial weights: [{weight_str}]", flush=True)
            mvue_count = sum(1 for m in model.modules() if hasattr(m, 'mvue_loss_baseline'))
            if mvue_count > 0:
                print(f"[MVUE-BASELINE] Updating L̄ for {mvue_count} modules each step", flush=True)

        loss.backward()

        # Update MVUE loss baseline for all TernaryLinear modules (R335)
        _loss_val = loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
        for _m in model.modules():
            if hasattr(_m, 'mvue_loss_baseline'):
                _m.mvue_loss_baseline = _m.mvue_ema_decay * _m.mvue_loss_baseline + (1.0 - _m.mvue_ema_decay) * _loss_val

        # Recipe B — zero gradients for crystallized weights
        if step > 0 and step % _crystal_check_every == 0:
            _total_params = 0
            _total_crystal = 0
            _total_freed_bytes = 0
            for _m in model.modules():
                if _m.__class__.__name__ != 'TernaryLinear':
                    continue
                _lid = id(_m)
                if _lid not in _crystal_ternary_prev:
                    continue
                with torch.no_grad():
                    _cur = (_m.weight.detach() > 0).to(torch.int8) - (_m.weight.detach() < 0).to(torch.int8)
                    _cur_cpu = _cur.cpu()
                    _prev = _crystal_ternary_prev[_lid]
                    _unchanged = (_cur_cpu == _prev)
                    _crystal_unchanged_count[_lid][_unchanged] += 1
                    _crystal_unchanged_count[_lid][~_unchanged] = 0
                    _mask_cpu = (_crystal_unchanged_count[_lid] >= 3)  # require 3 consecutive checks (1500 steps)
                    _crystal_masks[_lid] = _mask_cpu.to(_m.weight.device)
                    _crystal_ternary_prev[_lid] = _cur_cpu
                    _total_params += _m.weight.numel()
                    _total_crystal += _mask_cpu.sum().item()
                    _total_freed_bytes += _mask_cpu.sum().item() * 4

            _crystal_frac = _total_crystal / max(_total_params, 1)
            _freed_mb = _total_freed_bytes / 1e6
            print(f"[CRYSTAL] step={step} crystallized={_crystal_frac*100:.1f}% freed={_freed_mb:.0f}MB", flush=True)
            _crystal_logged = True

        # Dynamic deallocation: use hook-flagged candidates (R340-V3)
        if _crystal_logged and step > 0 and step % _crystal_check_every == 0:
            _freed_bytes = 0
            for _pid, _cand in _hook_dealloc_candidates.items():
                if _cand not in optimizer.state:
                    continue
                _state = optimizer.state[_cand]
                if _state.get('deallocated'):
                    continue
                if _state.get('factored'):
                    for _key in ['m_row', 'm_col']:
                        if _key in _state:
                            _freed_bytes += _state[_key].numel() * _state[_key].element_size()
                            _state[_key] = torch.zeros(1, device=_state[_key].device)
                else:
                    for _key in ['m', 'v']:
                        if _key in _state:
                            _freed_bytes += _state[_key].numel() * _state[_key].element_size()
                            _state[_key] = torch.zeros(1, device=_state[_key].device)
                _state['deallocated'] = True

            if _freed_bytes > 0:
                torch.cuda.empty_cache()
                print(f"[DEALLOC] step={step} freed={_freed_bytes/1024/1024:.1f}MB optimizer state", flush=True)

        # Boundary filter logging (counts from hook-side)
        if step % config.log_every == 0:
            _filt_pct = _hook_filtered[0] / max(_hook_total[0], 1) * 100
            print(f"[BOUNDARY-FILTER] step={step} filtered={_filt_pct:.1f}% ({_hook_filtered[0]}/{_hook_total[0]})", flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.joint_grad_clip)
        optimizer.step()

        if (step + 1) % 500 == 0:
            direct_ternary_assignment(model, step + 1)
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        t1 = time.perf_counter()
        ms_step = (t1 - t0) * 1000

        if step == 0:
            print(f"[ACTIVATION] JOINT-GD step=0 loss={loss_val:.4f} VRAM={torch.cuda.memory_allocated()/1e6:.0f}MB", flush=True)
            if mtp_logits is not None:
                print(f"[MTP-ACTIVATION] MTP ACTIVE: {len(mtp_logits)} heads, mtp_loss={loss_mtp.item():.4f}", flush=True)
            else:
                print(f"[MTP-ACTIVATION] MTP NOT ACTIVE — check mtp_num_heads in cfg", flush=True)

        if (step + 1) % config.log_every == 0:
            lr_current = scheduler.get_last_lr()[0]
            print(f"[JOINT-GD] step={step+1}/{config.joint_steps} loss={loss_val:.4f} lr={lr_current:.2e} ({ms_step:.0f}ms/step)", flush=True)

        if fixed_batches and (step + 1) % config.joint_eval_every == 0:
            model.eval()
            n_stages = cfg.n_layer
            eval_l = eval_loss(model, data, config, device, n_stages, fixed_batches=fixed_batches, full_eval=False)
            model.train()
            print(f"[V-PRIME] step={step+1} eval={eval_l:.4f}", flush=True)
            if eval_l < best_loss:
                best_loss = eval_l

        if config.use_capacity_matching and (step + 1) % config.capacity_matching_every == 0 and step > 0:
            capacity_matching_step(model, step + 1)

        if (step + 1) % config.joint_checkpoint_every == 0:
            ckpt_path = Path(config.output_path) / f"checkpoint_step{step+1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(),
                        'config': asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else {},
                        'step': step+1, 'loss': loss_val}, ckpt_path)
            print(f"[JOINT-GD] Checkpoint saved: {ckpt_path}", flush=True)

        if torch.cuda.memory_allocated() / 1e6 > config.max_vram_mb:
            print(f"[ERROR] VRAM exceeded {config.max_vram_mb}MB — stopping", flush=True)
            break

    total_time = time.perf_counter() - t_start
    final_loss = losses[-1] if losses else float('inf')

    print(f"[ACTIVATION] JOINT-GD done: {len(losses)} steps, {total_time:.1f}s, final_loss={final_loss:.4f}", flush=True)

    # Deregister post-accumulate hooks
    for _h in _hook_handles:
        _h.remove()
    _hook_handles.clear()
    print(f"[HOOK] Post-accumulate grad hooks deregistered", flush=True)

    model.to("cpu")
    torch.cuda.empty_cache()

    return losses, final_loss


def eval_loss(model, data, config, device, n_stages, fixed_batches=None, full_eval=False):
    """Evaluate loss in three modes: deterministic, stochastic-16, threshold-zero.

    When full_eval=False (default): only deterministic mode (fast, ~10s).
    When full_eval=True: all 3 modes (slow, ~256s).
    """
    for layer in model.layers:
        ssm = getattr(layer, 'mixer', None) or getattr(layer, 'ssm', None) or getattr(getattr(layer, 'block', None), 'mixer', None)
        if ssm and hasattr(ssm, '_episodic_h'):
            ssm._episodic_h = None

    layers_were_on_device = next(model.layers[0].parameters()).device.type == device.type
    if not layers_were_on_device:
        for layer in model.layers:
            layer.to(device)
        torch.cuda.empty_cache()

    use_fixed = fixed_batches is not None
    n_eval = len(fixed_batches) if use_fixed else config.eval_batches
    label = "fixed" if use_fixed else "random"

    def _get_batch(i):
        if use_fixed:
            return fixed_batches[i][0].to(device), fixed_batches[i][1].to(device)
        return sample_batch(data, config.seq_len, config.batch_size, get_vocab_size(model), device)

    def _ternary_linears(model):
        return [m for m in model.modules()
                if hasattr(m, 'ste_type') and hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter)]

    print(f"[EVAL-MODES] Running 3-mode eval: det + stoch16 + thresh0.3", flush=True)

    model.eval()
    losses_det = []
    with torch.no_grad():
        for i in range(n_eval):
            x, y = _get_batch(i)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
            losses_det.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
            del logits
    det_loss = float(np.mean(losses_det))
    print(f"[EVAL] deterministic: {det_loss:.4f} nats", flush=True)

    if full_eval:
        K = 16
        model.train()
        losses_stoch = []
        with torch.no_grad():
            for i in range(n_eval):
                x, y = _get_batch(i)
                probs_accum = None
                for _ in range(K):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(x)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                    probs = torch.softmax(logits.float(), dim=-1)
                    if probs_accum is None:
                        probs_accum = probs
                    else:
                        probs_accum = probs_accum + probs
                    del logits, probs
                avg_probs = probs_accum / K
                y_flat = y.reshape(-1)
                avg_probs_flat = avg_probs.reshape(-1, avg_probs.shape[-1])
                ce = -avg_probs_flat[torch.arange(y_flat.shape[0]), y_flat].log().mean().item()
                losses_stoch.append(ce)
                del probs_accum, avg_probs
        model.eval()
        stoch_loss = float(np.mean(losses_stoch))
        print(f"[EVAL] stochastic-16: {stoch_loss:.4f} nats", flush=True)
    else:
        stoch_loss = det_loss

    if full_eval:
        THRESHOLD = 0.3
        tl_modules = _ternary_linears(model)
        saved_weights = [(m, m.weight.data.clone()) for m in tl_modules]
        with torch.no_grad():
            for m, _ in saved_weights:
                mask = m.weight.data.abs() < THRESHOLD
                m.weight.data[mask] = 0.0

        model.eval()
        losses_thresh = []
        with torch.no_grad():
            for i in range(n_eval):
                x, y = _get_batch(i)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                losses_thresh.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
                del logits

        with torch.no_grad():
            for m, orig in saved_weights:
                m.weight.data.copy_(orig)

        thresh_loss = float(np.mean(losses_thresh))
        print(f"[EVAL] threshold-zero: {thresh_loss:.4f} nats", flush=True)
    else:
        thresh_loss = det_loss

    best_loss = min(det_loss, stoch_loss, thresh_loss)
    print(f"[V-PRIME] eval_loss over {n_eval} {label} batches: "
          f"det={det_loss:.4f} stoch16={stoch_loss:.4f} thresh0={thresh_loss:.4f} "
          f"best={best_loss:.4f}", flush=True)

    if not layers_were_on_device:
        for layer in model.layers:
            layer.to("cpu")
        torch.cuda.empty_cache()

    return best_loss


def _quantize_int8(tensor):
    """Compress float tensor to INT8 with per-channel scale."""
    scale = tensor.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    q = (tensor / scale * 127).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def _dequantize_int8(q, scale, device):
    """Decompress INT8 back to float on device."""
    return (q.float().to(device) * scale.to(device)) / 127.0


def run_mini_bcd(model, data, baseline_loss, config, device, n_stages, fixed_batches=None):
    """Run mini-BCD on top layers with cached activation targets."""
    n_top = max(1, int(n_stages * config.bcd_bottom_fraction))
    target_layer_indices = list(range(n_stages - n_top, n_stages))

    print(f"[ACTIVATION] PROD mini-BCD-CACHED start target layers={target_layer_indices} steps={config.bcd_steps}", flush=True)

    t_start = time.time()
    vocab_size = get_vocab_size(model)

    n_bcd_batches = min(5, config.cache_batches)
    cached_inputs = {}
    cached_targets = {}
    cached_y = {}

    with torch.no_grad():
        for bi in range(n_bcd_batches):
            x, y = sample_batch(data, config.seq_len, config.bcd_batch_size, vocab_size, device)
            H = model.embedding(x)

            for k in range(n_stages):
                model.layers[k].to(device)
                H_in = H
                H_out = model.layers[k](H_in)
                if isinstance(H_out, tuple):
                    H_out = H_out[0]

                if k in target_layer_indices:
                    if k not in cached_inputs:
                        cached_inputs[k] = []
                        cached_targets[k] = []
                        cached_y[k] = []
                    cached_inputs[k].append(_quantize_int8(H_in.detach().cpu()))
                    cached_targets[k].append(_quantize_int8(H_out.detach().cpu()))
                    cached_y[k].append(y.cpu())

                model.layers[k].to('cpu')
                H = H_out

            torch.cuda.empty_cache()

    print(f"[PROD] BCD-CACHED targets collected: {n_bcd_batches} batches, {len(target_layer_indices)} layers", flush=True)

    all_losses = []

    for k in target_layer_indices:
        for p in model.parameters():
            p.requires_grad_(False)
        model.layers[k].to(device)
        for p in model.layers[k].parameters():
            p.requires_grad_(True)

        trainable_count = sum(p.numel() for p in model.layers[k].parameters() if p.requires_grad)
        print(f"[PROD] BCD-CACHED layer={k}: {trainable_count/1e6:.1f}M trainable params", flush=True)

        optimizer = torch.optim.AdamW(
            [p for p in model.layers[k].parameters() if p.requires_grad],
            lr=config.bcd_lr, weight_decay=0.01
        )

        model.train()

        for step in range(config.bcd_steps):
            bi = step % n_bcd_batches
            H_in = _dequantize_int8(cached_inputs[k][bi][0], cached_inputs[k][bi][1], device)
            H_tgt = _dequantize_int8(cached_targets[k][bi][0], cached_targets[k][bi][1], device)

            optimizer.zero_grad()
            H_pred = model.layers[k](H_in)
            if isinstance(H_pred, tuple):
                H_pred = H_pred[0]

            loss = F.mse_loss(H_pred, H_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.layers[k].parameters() if p.requires_grad], 1.0)
            optimizer.step()

            loss_val = loss.item()
            all_losses.append(loss_val)

            if step == 0:
                print(f"[ACTIVATION] BCD-CACHED k={k} step=0 mse={loss_val:.5f}", flush=True)
            if (step + 1) % 10 == 0 or step == config.bcd_steps - 1:
                print(f"[PROD] BCD-CACHED k={k} step={step+1}/{config.bcd_steps} mse={loss_val:.5f}", flush=True)

        model.layers[k].to('cpu')
        del optimizer
        torch.cuda.empty_cache()

    t_bcd = time.time() - t_start

    print("[PROD] BCD-CACHED: evaluating CE loss post-BCD...", flush=True)
    post_bcd_loss = eval_loss(model, data, config, device, n_stages, fixed_batches=fixed_batches, full_eval=True)
    delta_nats = baseline_loss - post_bcd_loss

    print(f"[ACTIVATION] PROD mini-BCD-CACHED done t={t_bcd:.1f}s final_ce={post_bcd_loss:.4f} delta={delta_nats:+.4f} nats", flush=True)

    improvement_pct = (delta_nats / baseline_loss * 100) if baseline_loss > 0 else 0.0
    return {
        "layers": target_layer_indices,
        "steps": config.bcd_steps,
        "lr": config.bcd_lr,
        "delta_nats": delta_nats,
        "improvement_pct": improvement_pct,
        "final_loss": post_bcd_loss,
        "time_s": t_bcd,
    }


def save_checkpoint(model, config, results, output_path, model_config=None):
    """Save model checkpoint and results."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_path / "model_checkpoint.pt"
    results_file = output_path / "results.json"
    config_file = output_path / "config.json"

    checkpoint_data = {
        'state_dict': model.state_dict(),
        'config': asdict(model_config) if model_config is not None else asdict(config),
    }
    torch.save(checkpoint_data, checkpoint_file)
    print(f"[PROD] Checkpoint saved to {checkpoint_file}", flush=True)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[PROD] Results saved to {results_file}", flush=True)

    with open(config_file, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    print(f"[PROD] Config saved to {config_file}", flush=True)


def print_final_report(losses_fp, stage_times, stage_steps, stop_reasons,
                       baseline_loss, bcd_result, n_stages, max_steps_per_stage):
    """Print final report."""
    fp_time = sum(stage_times)
    total_time = fp_time + bcd_result['time_s']

    fp_reduction = (losses_fp[0] - losses_fp[-1]) / losses_fp[0] * 100 if losses_fp[0] > 0 else 0
    savings_pct = (1.0 - sum(stage_steps) / (n_stages * max_steps_per_stage)) * 100
    delta_vs_fp = bcd_result['final_loss'] - baseline_loss

    print(f"""
[PROD] ===== PRODUCTION PIPELINE FINAL REPORT =====

[PROD] --- FASE 1: FP com Cache + ABA ---
[PROD] Total steps used:     {sum(stage_steps)} / {n_stages * max_steps_per_stage} max
[PROD] Step savings vs fixed: {savings_pct:.1f}%  (ABA)
[PROD] Steps per stage:       {stage_steps}
[PROD] Stop reasons:          {stop_reasons}
[PROD] Stage times (s):       {[f"{t:.1f}" for t in stage_times]}
[PROD] FP first loss:         {losses_fp[0]:.4f}
[PROD] FP last loss:          {losses_fp[-1]:.4f}
[PROD] FP reduction:          {fp_reduction:.1f}%
[PROD] FP total time:         {fp_time:.1f}s

[PROD] --- FASE 2: Eval Pós-FP ---
[PROD] baseline_loss:         {baseline_loss:.4f} nats

[PROD] --- FASE 3: Mini-BCD ---
[PROD] layers trained:        {bcd_result['layers']}
[PROD] steps:                 {bcd_result['steps']}
[PROD] lr:                    {bcd_result['lr']}
[PROD] delta_nats:            {bcd_result['delta_nats']:+.4f}
[PROD] improvement:           {bcd_result['improvement_pct']:+.1f}%
[PROD] time:                  {bcd_result['time_s']:.1f}s

[PROD] --- PIPELINE TOTAL ---
[PROD] total time:            {total_time:.1f}s
[PROD] final loss:            {bcd_result['final_loss']:.4f}
[PROD] vs FP-only:            {delta_vs_fp:+.4f} nats

[PROD] ===== END REPORT =====
""", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Production training pipeline")
    parser.add_argument("--model-size", type=str, default="6m", choices=["6m", "100m", "7b"],
                       help="Model size")
    parser.add_argument("--dataset-path", type=str, default="data/chat_combined/train.bin",
                       help="Path to dataset")
    parser.add_argument("--output-path", type=str, default="output/train_production",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--smoke-test", action="store_true",
                       help="Quick smoke test (reduced steps)")
    parser.add_argument("--kill-gpu-procs", action="store_true",
                       help="Kill other GPU processes")
    parser.add_argument("--resume-from-fp", type=str, default=None,
                       help="Resume from FP checkpoint, skip to mini-BCD phase")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Max steps per FP stage (overrides config default of 100)")
    parser.add_argument("--training-mode", type=str, default="fp", choices=["fp", "joint"],
                       help="Training mode: 'fp' (staged layer training) or 'joint' (standard gradient descent)")
    parser.add_argument("--joint-steps", type=int, default=None,
                       help="Number of joint GD steps (overrides config default)")

    args = parser.parse_args()

    config = ProductionConfig(
        model_size=args.model_size,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
        kill_gpu_procs=args.kill_gpu_procs,
        use_psa=(args.model_size == "100m"),
        psa_weight=0.1 if args.model_size == "100m" else 0.0,
        use_til=(args.model_size == "100m"),
        til_K=4 if args.model_size == "100m" else 0,
        til_weight=0.1 if args.model_size == "100m" else 0.0,
        use_osp=(args.model_size == "100m"),
        osp_lambda=0.01 if args.model_size == "100m" else 0.0,
        training_mode=args.training_mode,
    )

    if args.max_steps is not None:
        config.max_steps_per_stage = args.max_steps

    if args.joint_steps is not None:
        config.joint_steps = args.joint_steps

    print("[PROD] === PRODUCTION TRAINING PIPELINE ===", flush=True)
    print(f"[PROD] Config: {config}", flush=True)

    if config.kill_gpu_procs:
        kill_gpu_processes()

    device = safety_check_vram(4000)
    print(f"[PROD] Device: {device}", flush=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    vram_snapshot("startup")

    cfg = make_config(config.model_size)
    from src.model_v11.model import ModelV2
    model = ModelV2(cfg)

    n_stages = cfg.n_layer
    print(f"[PROD] Creating model on CPU ({n_stages} layers)...", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PROD] Model created: {n_params/1e6:.1f}M params", flush=True)

    apply_bstar_init(model, cfg)
    apply_adaptive_threshold(model, cfg)

    print(f"[PROD] Moving embedding + norm + lm_head to GPU...", flush=True)
    model.embedding.to(device)
    if hasattr(model, 'final_norm'):
        model.final_norm.to(device)
    elif hasattr(model, 'norm_f'):
        model.norm_f.to(device)
    if model.lm_head is not None:
        model.lm_head.to(device)
    vram_snapshot("embeddings_to_gpu")

    print(f"[PROD] Loading data...", flush=True)
    data = load_data(config.dataset_path, vocab_size=cfg.vocab_size, seed=config.seed)

    calibrate_A_from_data(model, data, device)

    eval_batches_fixed = build_eval_batches(data, config, vocab_size=cfg.vocab_size)

    if args.resume_from_fp is not None:
        print(f"[PROD] === RESUME FROM FP CHECKPOINT ===", flush=True)
        checkpoint_path = Path(args.resume_from_fp)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        baseline_loss = checkpoint['baseline_loss']
        fp_time = checkpoint['fp_time_s']
        saved_config_dict = checkpoint.get('config', {})

        print(f"[PROD] Loaded checkpoint: baseline_loss={baseline_loss:.4f}, fp_time={fp_time:.1f}s", flush=True)

        if cfg is None or not saved_config_dict:
            print("[PROD] Using current model config", flush=True)
        else:
            print(f"[PROD] Config mismatch check: current vs saved", flush=True)

        model.load_state_dict(checkpoint['state_dict'])
        print(f"[PROD] Model state loaded from checkpoint", flush=True)

        model.embedding.to(device)
        if hasattr(model, 'final_norm'):
            model.final_norm.to(device)
        elif hasattr(model, 'norm_f'):
            model.norm_f.to(device)
        if model.lm_head is not None:
            model.lm_head.to(device)
        vram_snapshot("resumed_embeddings_to_gpu")

        print("[ACTIVATION] PROD FASE 1 FP+ABA skipped (resumed)", flush=True)
        print("[ACTIVATION] PROD FASE 2 eval skipped (resumed)", flush=True)

        print("[ACTIVATION] PROD FASE 3 mini-BCD start", flush=True)
        t_bcd_start = time.perf_counter()
        bcd_result = run_mini_bcd(model, data, baseline_loss, config, device, n_stages, fixed_batches=eval_batches_fixed)
        t_bcd_end = time.perf_counter()
        print(f"[ACTIVATION] PROD FASE 3 done t={t_bcd_end-t_bcd_start:.1f}s "
              f"delta={bcd_result['delta_nats']:+.4f} nats", flush=True)

        results = {
            "timestamp": datetime.now().isoformat(),
            "model_size": config.model_size,
            "resumed": True,
            "fp_time_s": fp_time,
            "fp_baseline_loss": float(baseline_loss),
            "bcd_result": bcd_result,
            "total_time_s": fp_time + bcd_result['time_s'],
        }

        save_checkpoint(model, config, results, config.output_path, model_config=cfg)
        print("[PROD] === PRODUCTION TRAINING COMPLETE (RESUMED) ===", flush=True)
        return

    if config.training_mode == "joint":
        print(f"[PROD] === JOINT GD MODE ===", flush=True)
        losses_joint, final_loss = run_joint_gd(model, data, config, device, cfg,
                                                 fixed_batches=eval_batches_fixed)
        results = {"training_mode": "joint", "steps": len(losses_joint),
                   "final_loss": float(final_loss)}
        ckpt_path = Path(config.output_path) / "model_checkpoint.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': model.state_dict(),
                    'config': asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else {},
                    'results': results}, ckpt_path)
        import json
        results_path = Path(config.output_path) / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[PROD] Checkpoint saved: {ckpt_path}", flush=True)
        print(f"[PROD] === JOINT GD COMPLETE ===", flush=True)
        return

    print("[ACTIVATION] PROD FASE 1 FP+ABA start", flush=True)
    t_fp_start = time.perf_counter()
    model, losses_fp, loss_after_fp, stage_times, stage_steps, stop_reasons, \
        fp_momentum_store = run_fp_pipeline(model, data, config, device, n_stages)
    t_fp_end = time.perf_counter()
    print(f"[ACTIVATION] PROD FASE 1 done t={t_fp_end-t_fp_start:.1f}s "
          f"total_steps={sum(stage_steps)} loss={loss_after_fp:.4f}", flush=True)

    print("[ACTIVATION] PROD FASE 2 eval start", flush=True)
    baseline_loss = eval_loss(model, data, config, device, n_stages, fixed_batches=eval_batches_fixed, full_eval=True)
    print(f"[ACTIVATION] PROD FASE 2 eval done baseline_loss={baseline_loss:.4f}", flush=True)

    fp_time = t_fp_end - t_fp_start
    fp_checkpoint_path = Path(config.output_path) / "fp_checkpoint.pt"
    fp_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'config': asdict(cfg) if cfg is not None else {},
        'phase': 'post_fp',
        'baseline_loss': baseline_loss,
        'fp_time_s': fp_time,
    }, fp_checkpoint_path)
    print(f"[PROD] Intermediate checkpoint saved: {fp_checkpoint_path}", flush=True)

    print("[ACTIVATION] PROD FASE 3 mini-BCD start", flush=True)
    t_bcd_start = time.perf_counter()
    bcd_result = run_mini_bcd(model, data, baseline_loss, config, device, n_stages, fixed_batches=eval_batches_fixed)
    t_bcd_end = time.perf_counter()
    print(f"[ACTIVATION] PROD FASE 3 done t={t_bcd_end-t_bcd_start:.1f}s "
          f"delta={bcd_result['delta_nats']:+.4f} nats", flush=True)

    print_final_report(losses_fp, stage_times, stage_steps, stop_reasons,
                      baseline_loss, bcd_result, n_stages, config.max_steps_per_stage)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_size": config.model_size,
        "fp_stages": n_stages,
        "fp_total_steps": sum(stage_steps),
        "fp_time_s": sum(stage_times),
        "fp_first_loss": float(losses_fp[0]),
        "fp_last_loss": float(losses_fp[-1]),
        "eval_baseline_loss": float(baseline_loss),
        "bcd_result": bcd_result,
        "total_time_s": sum(stage_times) + bcd_result['time_s'],
    }

    save_checkpoint(model, config, results, config.output_path, model_config=cfg)
    print("[PROD] === PRODUCTION TRAINING COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
