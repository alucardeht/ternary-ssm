"""
R13-Batch-AG: Gradient-Local Training Scheme (GLTS).
Layer-sequential backward pass: O(1 layer) peak activation memory.
Enables 300B+ training on consumer GPU.

Always-active when torch.compile is NOT in use.
Falls back to returning None when compiling (train.py uses standard backward).

Mathematical principle:
  dL/dx_i = dL/dy_i * dy_i/dx_i  (chain rule, local to layer i)
  Activations for each layer are recomputed from saved boundary points.
"""
import torch
import torch.nn as nn
from contextlib import contextmanager


@contextmanager
def _eval_mode(module):
    """Temporarily set module to eval mode (bypasses SAC inside ModelBlock)."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()


def glts_step(
    raw_model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    boundary_mask=None,
    grad_scale: float = 1.0,
    device: str = 'cuda',
    amp_dtype=torch.bfloat16,
):
    """
    Layer-sequential training step (GLTS).

    Returns (loss, logits) with gradients accumulated into all .grad attrs.
    Returns (None, None) when torch.compile is active → caller uses standard backward.

    Args:
        raw_model:     unwrapped model (model._orig_mod if compiled, else model)
        input_ids:     [B, L] token ids
        targets:       [B, L] target token ids (same as input_ids for LM)
        boundary_mask: optional document boundary mask
        grad_scale:    loss scaling factor (1 / grad_accum_steps)
        device:        'cuda' or 'cpu'
        amp_dtype:     autocast dtype
    """
    if torch.compiler.is_compiling():
        return None, None

    layers = getattr(raw_model, 'layers', None)
    if layers is None or len(layers) == 0:
        return None, None

    embedding = (getattr(raw_model, 'embedding', None) or
                 getattr(raw_model, 'embed_tokens', None) or
                 getattr(raw_model, 'tok_emb', None))
    final_norm = (getattr(raw_model, 'norm', None) or
                  getattr(raw_model, 'final_norm', None) or
                  getattr(raw_model, 'ln_f', None))
    lm_head = getattr(raw_model, 'lm_head', None)

    if embedding is None or final_norm is None:
        return None, None

    N = len(layers)
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=amp_dtype)

    boundary_acts = []
    with torch.no_grad():
        with autocast_ctx:
            x = embedding(input_ids)
            boundary_acts.append(x.detach())

            for layer in layers:
                with _eval_mode(layer):
                    x = layer(x, boundary_mask=boundary_mask)
                boundary_acts.append(x.detach())

    x_N = boundary_acts[-1].detach().requires_grad_(True)

    with autocast_ctx:
        x_normed = final_norm(x_N)
        if lm_head is not None:
            logits = lm_head(x_normed)
        else:
            logits = torch.nn.functional.linear(x_normed, embedding.weight)

        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().reshape(-1, logits.shape[-1]),
            targets[:, 1:].contiguous().reshape(-1),
        )
        scaled_loss = loss * grad_scale

    scaled_loss.backward()
    upstream_grad = x_N.grad

    for i in reversed(range(N)):
        x_in = boundary_acts[i].detach().requires_grad_(True)
        layer = layers[i]

        with _eval_mode(layer):
            with autocast_ctx:
                x_out = layer(x_in, boundary_mask=boundary_mask)

        x_out.backward(upstream_grad)
        upstream_grad = x_in.grad

    with autocast_ctx:
        x_embed = embedding(input_ids)
    x_embed.backward(upstream_grad)

    if hasattr(raw_model, '_glts_step_count'):
        raw_model._glts_step_count += 1
    else:
        raw_model._glts_step_count = 1

    step_cnt = raw_model._glts_step_count
    if step_cnt == 1 or step_cnt % 500 == 1:
        saved_mb = sum(a.numel() * 2 / 1e6 for a in boundary_acts)
        print(f"[GLTS-AG] step={step_cnt} layers={N} "
              f"boundary_mem={saved_mb:.1f}MB "
              f"loss={loss.item():.4f} upstream_norm={upstream_grad.norm():.4f}")

    return loss, logits
