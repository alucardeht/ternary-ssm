import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from .model import ModelV2
from .components.ssm import SelectiveSSM, SelectiveSSMSSD
from .components.attention import SlidingWindowAttention
from .components.rwkv7 import RWKV7Block


_fused_ssm_kernel = None

def get_fused_ssm_kernel():
    global _fused_ssm_kernel
    if _fused_ssm_kernel is None:
        try:
            from .kernels.fused_ssm_wrapper import FusedSSMKernel
            _fused_ssm_kernel = FusedSSMKernel()
        except Exception:
            _fused_ssm_kernel = False
    return _fused_ssm_kernel if _fused_ssm_kernel is not False else None


@dataclass
class InferenceState:
    layer_states: List[Dict[str, Any]]
    v_first: Optional[torch.Tensor]
    position: int


def init_state(model: ModelV2, batch_size: int, device: torch.device) -> InferenceState:
    model_dtype = next(model.parameters()).dtype
    # Accumulator states (SSM h, RWKV7 state) always use float32 for numerical
    # stability during sequential accumulation. Buffer states (conv_state,
    # k/v cache, x_prev) use model_dtype to avoid dtype mismatches with model outputs.
    accum_dtype = torch.float32
    buffer_dtype = model_dtype
    layer_states = []

    for layer in model.layers:
        block = layer.block
        state_dict = {}

        if isinstance(block, SelectiveSSMSSD):
            d_inner = block.d_inner
            n_heads = block.n_heads
            d_state = block.d_state
            head_dim = d_inner // n_heads

            state_dict['h'] = torch.zeros(
                batch_size, n_heads, d_state, head_dim,
                device=device, dtype=accum_dtype
            )
            state_dict['conv_state'] = torch.zeros(
                batch_size, d_inner, block.d_conv,
                device=device, dtype=buffer_dtype
            )

        elif isinstance(block, SelectiveSSM):
            d_inner = block.d_inner
            d_state = block.d_state

            state_dict['h'] = torch.zeros(
                batch_size, d_inner, d_state,
                device=device, dtype=accum_dtype
            )
            state_dict['conv_state'] = torch.zeros(
                batch_size, d_inner, block.d_conv,
                device=device, dtype=buffer_dtype
            )

        elif isinstance(block, SlidingWindowAttention):
            window_size = block.window_size
            n_heads = block.n_heads
            head_dim = block.head_dim

            state_dict['k_cache'] = torch.zeros(
                batch_size, n_heads, window_size, head_dim,
                device=device, dtype=buffer_dtype
            )
            state_dict['v_cache'] = torch.zeros(
                batch_size, n_heads, window_size, head_dim,
                device=device, dtype=buffer_dtype
            )
            state_dict['cache_len'] = 0

        elif getattr(layer, 'is_rwkv7', False):
            d_model = block.d_model
            n_heads = block.n_heads
            head_size = block.head_size

            state_dict['state'] = torch.zeros(
                batch_size, n_heads, head_size, head_size,
                device=device, dtype=accum_dtype
            )
            state_dict['x_prev'] = torch.zeros(
                batch_size, d_model,
                device=device, dtype=buffer_dtype
            )

        layer_states.append(state_dict)

    v_first = None
    if any(getattr(layer, 'is_rwkv7', False) for layer in model.layers):
        v_first = torch.zeros(
            batch_size, model.config.d_model,
            device=device, dtype=buffer_dtype
        )

    return InferenceState(
        layer_states=layer_states,
        v_first=v_first,
        position=0
    )


def ssm_ssd_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    h: torch.Tensor,
    conv_state: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SSD single-step update (multi-head state space duality).

    Args:
        layer: ModelBlock with SelectiveSSMSSD
        x_t: (batch, d_model) - single token embedding
        h: (batch, n_heads, d_state, head_dim) - hidden state
        conv_state: (batch, d_inner, d_conv) - convolution state

    Returns:
        output: (batch, d_model)
        h_new: (batch, n_heads, d_state, head_dim)
        conv_state_new: (batch, d_inner, d_conv)
    """
    block = layer.block
    batch_size = x_t.shape[0]

    xz = block.in_proj(x_t)
    x_inner, z = xz.chunk(2, dim=-1)

    # Update convolution state
    x_inner_expanded = x_inner.unsqueeze(-1)  # (batch, d_inner, 1)
    conv_state_new = torch.cat(
        [conv_state[:, :, 1:], x_inner_expanded],
        dim=-1
    )
    x_conv = (conv_state_new * block.conv1d.weight.squeeze(1)).sum(dim=-1) + block.conv1d.bias
    x_conv = F.silu(x_conv)  # (batch, d_inner)

    # Project to get dt, B, C
    x_dbc = block.x_proj(x_conv)  # (batch, dt_rank + 2*d_state)
    dt_input = x_dbc[:, :block.dt_rank]  # (batch, dt_rank)
    B = x_dbc[:, block.dt_rank:block.dt_rank + block.d_state]  # (batch, d_state)
    C = x_dbc[:, block.dt_rank + block.d_state:]  # (batch, d_state)

    dt = F.softplus(block.dt_proj(dt_input))  # (batch, d_inner)
    A = -torch.exp(block.A_log)  # (n_heads,) - scalar per head

    # Reshape inputs for multi-head computation
    n_heads = block.n_heads
    head_dim = block.d_inner // n_heads

    # Average dt across head dimensions to get per-head scalar
    dt_heads = dt.view(batch_size, n_heads, head_dim).mean(dim=-1).float()  # (batch, n_heads)

    # Reshape x_conv to (batch, n_heads, head_dim)
    x_heads = x_conv.view(batch_size, n_heads, head_dim).float()  # (batch, n_heads, head_dim)

    # Try fused SSM kernel
    fused_kernel = get_fused_ssm_kernel()
    if fused_kernel is not None and fused_kernel.available:
        D_flat = block.D.view(-1).float() if hasattr(block, 'D') else torch.ones(
            block.d_inner, device=x_t.device, dtype=torch.float32
        )

        y, h_new = fused_kernel.step(
            h.float(),
            dt_heads,
            A.float(),
            B.float(),
            C.float(),
            x_heads,
            D_flat
        )
    else:
        # Compute dA: exp(dt_heads * A) where A is scalar per head
        # dt_heads: (batch, n_heads), A: (n_heads,)
        # Result: (batch, n_heads, 1)
        dA = torch.exp((dt_heads * A.unsqueeze(0)).unsqueeze(-1))  # (batch, n_heads, 1)

        # Compute dB * x: dt * B * x
        # dt_heads: (batch, n_heads, 1)
        # B: (batch, d_state) -> need to broadcast to (batch, n_heads, d_state)
        # x_heads: (batch, n_heads, head_dim)
        B_expanded = B.unsqueeze(1).expand(batch_size, n_heads, -1).float()  # (batch, n_heads, d_state)

        # dB_x = dt_heads * (B expanded) * x_heads
        # (batch, n_heads, 1) * (batch, n_heads, d_state, 1) * (batch, n_heads, 1, head_dim)
        dB_x = (dt_heads.unsqueeze(-1) * B_expanded.unsqueeze(-1) *
                x_heads.unsqueeze(2))  # (batch, n_heads, d_state, head_dim)

        # Update state: h_new = dA * h + dB_x
        # h: (batch, n_heads, d_state, head_dim)
        # dA: (batch, n_heads, 1)
        # dB_x: (batch, n_heads, d_state, head_dim)
        h_new = dA.unsqueeze(-1).float() * h.float() + dB_x.float()  # (batch, n_heads, d_state, head_dim)

        # Compute output: y = h_new @ C
        # C: (batch, d_state) -> (batch, 1, 1, d_state) for broadcasting
        # h_new: (batch, n_heads, d_state, head_dim)
        # Result should be (batch, n_heads, head_dim)
        C_expanded = C.unsqueeze(1).expand(batch_size, n_heads, -1).float()  # (batch, n_heads, d_state)
        y = (h_new * C_expanded.unsqueeze(-1)).sum(dim=2)  # (batch, n_heads, head_dim)

        # Reshape y back to (batch, d_inner)
        y = y.reshape(batch_size, block.d_inner)

        # Add residual
        y = y + x_conv * block.D  # (batch, d_inner)

    # Gate and project
    silu_z = F.silu(z)
    y = silu_z * y.to(x_t.dtype)

    output = layer.norm(x_t)
    output = output + block.out_proj(y)

    return output, h_new, conv_state_new



def ssm_mamba1_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    h: torch.Tensor,
    conv_state: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block = layer.block
    batch_size = x_t.shape[0]

    xz = block.in_proj(x_t)
    x_inner, z = xz.chunk(2, dim=-1)

    x_inner_expanded = x_inner.unsqueeze(-1)
    conv_state_new = torch.cat(
        [conv_state[:, :, 1:], x_inner_expanded],
        dim=-1
    )
    x_conv = (conv_state_new * block.conv1d.weight.squeeze(1)).sum(dim=-1) + block.conv1d.bias
    x_conv = F.silu(x_conv)

    x_dbc = block.x_proj(x_conv)
    dt_input = x_dbc[:, :block.dt_rank]
    B = x_dbc[:, block.dt_rank:block.dt_rank + block.d_state]
    C = x_dbc[:, block.dt_rank + block.d_state:]

    dt = F.softplus(block.dt_proj(dt_input))
    A = -torch.exp(block.A_log)

    x_conv_float = x_conv.to(torch.float32)
    dA = torch.exp(dt.unsqueeze(-1).to(torch.float32) * A.unsqueeze(0))
    dB_x = (dt.unsqueeze(-1).to(torch.float32) * B.unsqueeze(1).to(torch.float32)) * x_conv_float.unsqueeze(-1)

    h_new = dA * h + dB_x
    y = (h_new * C.unsqueeze(1).to(torch.float32)).sum(dim=-1).to(x_t.dtype) + x_conv * block.D

    silu_z = F.silu(z)
    y = silu_z * y

    output = layer.norm(x_t)
    output = output + block.out_proj(y)

    return output, h_new, conv_state_new


def swa_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: int,
    position: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    block = layer.block
    batch_size = x_t.shape[0]

    x_norm = block.attn_norm(x_t)

    q = block.q_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)
    k = block.k_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)
    v = block.v_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)

    cos_cached = block.rotary.cos_cached[:, :, position, :]
    sin_cached = block.rotary.sin_cached[:, :, position, :]

    def rotate_half(x_val):
        x_chunked = x_val.chunk(2, dim=-1)
        return torch.cat([-x_chunked[1], x_chunked[0]], dim=-1)

    q = q * cos_cached + rotate_half(q) * sin_cached
    k = k * cos_cached + rotate_half(k) * sin_cached

    cache_idx = cache_len % block.window_size
    k_cache[:, :, cache_idx, :] = k
    v_cache[:, :, cache_idx, :] = v

    cache_len_new = cache_len + 1
    valid_len = min(cache_len_new, block.window_size)

    scale = 1.0 / math.sqrt(block.head_dim)
    scores = torch.matmul(q.unsqueeze(-2), k_cache[:, :, :valid_len, :].transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)

    attn_out = torch.matmul(attn_weights, v_cache[:, :, :valid_len, :])
    attn_out = attn_out.squeeze(-2).view(batch_size, -1)

    attn_out = block.out_proj(attn_out)

    x = x_t + attn_out

    x_norm = block.mlp_norm(x)
    gate_out = F.silu(block.mlp_gate(x_norm)) * block.mlp_up(x_norm)
    mlp_out = block.mlp_down(gate_out)

    output = x + mlp_out

    return output, k_cache, v_cache, cache_len_new


def rwkv7_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    state: torch.Tensor,
    x_prev: torch.Tensor,
    v_first: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block = layer.block
    batch_size = x_t.shape[0]
    d_model = block.d_model

    x_norm = layer.norm(x_t)

    xx = x_prev - x_t

    xr = x_t + xx * block.x_r
    xk = x_t + xx * block.x_k
    xv = x_t + xx * block.x_v
    xw = x_t + xx * block.x_w
    xa = x_t + xx * block.x_a
    xg = x_t + xx * block.x_g

    r = block.W_r(xr).view(batch_size, block.n_heads, block.head_size)
    k = block.W_k(xk).view(batch_size, block.n_heads, block.head_size)
    v = block.W_v(xv).view(batch_size, block.n_heads, block.head_size)

    if v_first is None:
        v_first = v.detach()
    else:
        v_first_shaped = v_first.view(batch_size, block.n_heads, block.head_size)
        alpha = torch.sigmoid(block.v0 + (xv @ block.v1) @ block.v2).view(batch_size, block.n_heads, block.head_size)
        v = v + (v_first_shaped - v) * alpha
        v_first = v.detach()

    w = -F.softplus(-(block.w0 + (torch.tanh(xw @ block.w1) @ block.w2))) - 0.5
    w = torch.exp(w).view(batch_size, block.n_heads, block.head_size)

    a = torch.sigmoid(block.a0 + (xa @ block.a1) @ block.a2).view(batch_size, block.n_heads, block.head_size)
    g = (torch.sigmoid(xg @ block.g1) @ block.g2).view(batch_size, block.n_heads, block.head_size)

    kk = k * block.k_k.view(1, block.n_heads, block.head_size)
    kk = F.normalize(kk, dim=-1, p=2.0)
    k_scaled = k * (1 + (a - 1) * block.k_a.view(1, block.n_heads, block.head_size))

    r_mat = r.unsqueeze(-1)
    v_mat = v.unsqueeze(-1)
    w_mat = w.unsqueeze(-2)

    aa = (kk * a).unsqueeze(-1)
    bb = (-kk).unsqueeze(-2)
    aa_bb = aa @ bb

    vk = v_mat @ k_scaled.unsqueeze(-2)

    state = state * w_mat + state @ aa_bb + vk

    out = (state @ r_mat).squeeze(-1)
    out = out.view(batch_size, -1)

    out = block.group_norm(out.unsqueeze(1)).squeeze(1)

    r_flat = r.view(batch_size, -1)
    k_flat = k_scaled.view(batch_size, -1)
    v_flat = v.view(batch_size, -1)
    r_k = block.r_k.view(1, block.n_heads, block.head_size).expand_as(r).reshape(batch_size, -1)
    la_term = (r_flat * k_flat * r_k).sum(dim=-1, keepdim=True) * v_flat
    out = out + la_term

    output = block.W_o(out * g.view(batch_size, -1))

    return output, state, x_t, v_first


def model_step(
    model: ModelV2,
    token_id: torch.Tensor,
    state: InferenceState
) -> Tuple[torch.Tensor, InferenceState]:
    batch_size = token_id.shape[0]
    device = token_id.device

    x = model.embedding(token_id)
    x = x.squeeze(1) if x.dim() > 2 else x

    new_layer_states = []
    v_first = state.v_first

    for i, layer in enumerate(model.layers):
        block = layer.block
        layer_state = state.layer_states[i]

        if isinstance(block, SelectiveSSMSSD):
            h = layer_state['h']
            conv_state = layer_state['conv_state']

            output, h_new, conv_state_new = ssm_ssd_step(layer, x, h, conv_state)

            new_layer_states.append({
                'h': h_new,
                'conv_state': conv_state_new
            })
            x = output

        elif isinstance(block, SelectiveSSM):
            h = layer_state['h']
            conv_state = layer_state['conv_state']

            output, h_new, conv_state_new = ssm_mamba1_step(layer, x, h, conv_state)

            new_layer_states.append({
                'h': h_new,
                'conv_state': conv_state_new
            })
            x = output

        elif isinstance(block, SlidingWindowAttention):
            k_cache = layer_state['k_cache']
            v_cache = layer_state['v_cache']
            cache_len = layer_state['cache_len']

            output, k_cache_new, v_cache_new, cache_len_new = swa_step(
                layer, x, k_cache, v_cache, cache_len, state.position
            )

            new_layer_states.append({
                'k_cache': k_cache_new,
                'v_cache': v_cache_new,
                'cache_len': cache_len_new
            })
            x = output

        elif getattr(layer, 'is_rwkv7', False):
            state_mat = layer_state['state']
            x_prev = layer_state['x_prev']

            output, state_mat_new, x_prev_new, v_first = rwkv7_step(
                layer, x, state_mat, x_prev, v_first
            )

            new_layer_states.append({
                'state': state_mat_new,
                'x_prev': x_prev_new
            })
            x = output

    x = model.final_norm(x)

    if model.lm_head is not None:
        logits = model.lm_head(x)
    else:
        logits = F.linear(x, model.embedding.weight)

    logits = logits * model.output_multiplier
    logits = logits.unsqueeze(1)

    new_state = InferenceState(
        layer_states=new_layer_states,
        v_first=v_first,
        position=state.position + 1
    )

    return logits, new_state


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> int:
    logits = logits.squeeze(0).squeeze(0)

    if temperature == 0:
        return logits.argmax(dim=-1).item()

    logits = logits / temperature

    if top_k is not None:
        top_k_vals, top_k_indices = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered[top_k_indices] = top_k_vals
        logits = logits_filtered

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

    # Handle -inf values by masking them out, then apply softmax only to valid logits
    valid_mask = torch.isfinite(logits)
    if not valid_mask.any():
        # All logits are invalid, return uniform random
        return torch.randint(0, logits.shape[-1], (1,), device=logits.device).item()
    
    # Create valid logits tensor by clamping -inf values
    logits_safe = torch.where(valid_mask, logits, torch.tensor(float('-1e9'), device=logits.device, dtype=logits.dtype))
    probs = F.softmax(logits_safe, dim=-1)
    
    # Ensure probabilities are valid
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        # Fallback to uniform distribution
        probs = torch.ones_like(logits) / logits.shape[-1]
    
    # Ensure probabilities sum to 1
    probs = probs / probs.sum()
    
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id


@torch.no_grad()
def generate(
    model: ModelV2,
    prompt_ids: torch.Tensor,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    batch_size: int = 1
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device

    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    if prompt_ids.shape[0] < batch_size:
        prompt_ids = prompt_ids.expand(batch_size, -1)

    state = init_state(model, batch_size, device)

    for token_idx in range(prompt_ids.shape[1]):
        token = prompt_ids[:, token_idx:token_idx+1]
        _, state = model_step(model, token, state)

    seq = prompt_ids.clone()

    for _ in range(max_tokens):
        last_token = seq[:, -1:].clone()
        logits, state = model_step(model, last_token, state)

        next_token = sample_token(logits.squeeze(1), temperature, top_k, top_p)
        next_token = torch.tensor([[next_token]], device=device, dtype=torch.long)
        next_token = next_token.expand(batch_size, -1)

        seq = torch.cat([seq, next_token], dim=1)

    return seq
