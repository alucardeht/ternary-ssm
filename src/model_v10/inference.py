import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from .model import ModelV2
from .components.ssm import SelectiveSSM, SelectiveSSMSSD
from .components.attention import SlidingWindowAttention
from .components.rwkv7 import RWKV7Block
from .components.hierarchical_lm_head import HierarchicalLMHead
from .components.ternary import DEFAULT_THRESHOLD_COEFF


_fused_ssm_kernel = None
_triton_available = None
_triton_ssm_available = None
_fused_ssm_layer_available = None


def _get_triton_ssm():
    """Lazy import of Triton fused SSM recurrence kernel."""
    global _triton_ssm_available
    if _triton_ssm_available is None:
        try:
            from .kernels.triton.fused_ssm_recurrence import fused_ssm_recurrence_step
            _triton_ssm_available = fused_ssm_recurrence_step
        except Exception:
            _triton_ssm_available = False
    return _triton_ssm_available if _triton_ssm_available is not False else None


def _get_fused_ssm_layer():
    """Lazy import of Triton fused SSM layer mega-kernels (K1-K4)."""
    global _fused_ssm_layer_available
    if _fused_ssm_layer_available is None:
        try:
            from .kernels.triton import fused_ssm_layer as _mod
            _fused_ssm_layer_available = _mod
        except Exception:
            _fused_ssm_layer_available = False
    return _fused_ssm_layer_available if _fused_ssm_layer_available is not False else None


def _check_triton():
    """Lazy check for Triton availability."""
    global _triton_available
    if _triton_available is None:
        try:
            import triton
            _triton_available = True
        except ImportError:
            _triton_available = False
    return _triton_available

def get_fused_ssm_kernel():
    global _fused_ssm_kernel
    if _fused_ssm_kernel is None:
        try:
            from .kernels.fused_ssm_wrapper import FusedSSMKernel
            _fused_ssm_kernel = FusedSSMKernel()
        except Exception:
            _fused_ssm_kernel = False
    return _fused_ssm_kernel if _fused_ssm_kernel is not False else None


def prepare_for_inference(model, device='cuda', backend='auto', verbose=True, **kwargs):
    """Prepare model for optimized inference.

    Converts TernaryLinear layers to Triton GEMV kernels for
    matmul-free ternary inference (add/sub/skip operations only).

    Extra kwargs:
        hierarchical_lm (bool): Enable hierarchical LM head (default False).
        hlm_n_clusters (int): Number of vocab clusters (default 128).
        hlm_top_k_clusters (int): Clusters scored in fine pass (default 8).
    """
    info = {
        'backend': 'baseline',
        'n_converted': 0,
        'n_skipped': 0,
        'model_dtype': None,
    }

    model.eval()
    model.to(device)

    if kwargs.get('entropy_screening', False):
        _precompute_entropy_screening(model, verbose=verbose)
        es_k = kwargs.get('es_k_screen', 4096)
        model._es_k_screen = es_k
        info['entropy_screening'] = True
        info['es_k_screen'] = es_k

    use_triton = False
    if backend == 'triton':
        if device == 'cpu':
            raise ValueError("Triton backend requires CUDA device")
        if not _check_triton():
            raise ImportError("Triton not installed")
        use_triton = True
    elif backend == 'auto':
        use_triton = (str(device).startswith('cuda') and _check_triton())

    if use_triton:
        try:
            from .kernels.triton.ternary_gemv import convert_model_to_triton
            model, n_converted, n_skipped = convert_model_to_triton(
                model, device=device, verbose=verbose
            )
            if n_converted > 0:
                model.half()
                info['backend'] = 'triton'
                info['n_converted'] = n_converted
                info['n_skipped'] = n_skipped
                if verbose:
                    print(f"[Inference] Triton backend: {n_converted} layers converted, "
                          f"{n_skipped} skipped, model dtype=float16")

                # Prepare mega-kernel buffers for SSM layers
                if info.get('backend') == 'triton':
                    n_mk = _prepare_megakernel_bufs(model, device, verbose=verbose)
                    if n_mk > 0:
                        info['megakernel_layers'] = n_mk
            else:
                if verbose:
                    print("[Inference] No TernaryLinear layers found, using baseline")
        except Exception as e:
            if backend == 'triton':
                raise
            if verbose:
                print(f"[Inference] Triton conversion failed ({e}), using baseline")
    elif backend == 'tsar' or (backend == 'auto' and device == 'cpu'):
        if backend == 'tsar' and str(device).startswith('cuda'):
            raise ValueError("T-SAR backend is CPU-only")

        try:
            from .kernels.tsar_cpu import convert_model_to_tsar_cpu
            model, n_converted, n_skipped = convert_model_to_tsar_cpu(model, verbose=verbose)
            if n_converted > 0:
                info['backend'] = 'tsar'
                info['n_converted'] = n_converted
                info['n_skipped'] = n_skipped
                if verbose:
                    print(f"[Inference] T-SAR CPU backend: {n_converted} layers converted, {n_skipped} skipped")
            else:
                if backend == 'tsar':
                    if verbose:
                        print("[Inference] No convertible layers found for T-SAR, using baseline")
        except Exception as e:
            if backend == 'tsar':
                raise
            if verbose:
                print(f"[Inference] T-SAR conversion failed ({e}), using baseline")
    elif backend == 'popcount':
        try:
            from .kernels.popcount_gemv import convert_model_to_popcount
            model, n_converted, n_skipped = convert_model_to_popcount(
                model, device=str(device), verbose=verbose
            )
            if n_converted > 0:
                info['backend'] = 'popcount'
                info['n_converted'] = n_converted
                info['n_skipped'] = n_skipped
                if verbose:
                    print(f"[Inference] POPCOUNT backend: {n_converted} layers converted, "
                          f"{n_skipped} skipped (activation binarization enabled)")
            else:
                if verbose:
                    print("[Inference] No TernaryLinear layers found for POPCOUNT, using baseline")
        except Exception as e:
            raise  # popcount is explicit request, don't silently fall back
    elif backend == 'c_portable':
        try:
            from .kernels.c_portable.wrapper import convert_model_to_c_portable
            model, n_converted, n_skipped = convert_model_to_c_portable(model, verbose=verbose)
            if n_converted > 0:
                info['backend'] = 'c_portable'
                info['n_converted'] = n_converted
                info['n_skipped'] = n_skipped
                if verbose:
                    print(f"[Inference] C-Portable backend: {n_converted} layers converted, "
                          f"{n_skipped} skipped (scalar GEMV, portable)")
            else:
                if verbose:
                    print("[Inference] No TernaryLinear layers found, using baseline")
        except Exception as e:
            raise
    elif backend == 'wph':
        try:
            from .kernels.wph.wrapper import convert_model_to_wph
            model, n_converted, n_skipped = convert_model_to_wph(model, verbose=verbose)
            if n_converted > 0:
                info['backend'] = 'wph'
                info['n_converted'] = n_converted
                info['n_skipped'] = n_skipped
                if verbose:
                    print(f"[Inference] WPH backend: {n_converted} layers converted, "
                          f"{n_skipped} skipped (weight pattern hashing, 81 patterns)")
            else:
                if verbose:
                    print("[Inference] No TernaryLinear layers found, using baseline")
        except Exception as e:
            raise
    else:
        if verbose:
            reason = "CPU device" if device == 'cpu' else "Triton not available" if not _check_triton() else "baseline requested"
            print(f"[Inference] Baseline backend ({reason})")

    # --- Hierarchical LM Head (optional, reduces vocab projection bandwidth) ---
    if kwargs.get('hierarchical_lm', False):
        n_clusters = kwargs.get('hlm_n_clusters', 128)
        top_k = kwargs.get('hlm_top_k_clusters', 8)

        emb_weight = model.embedding.weight.data
        hlm = HierarchicalLMHead(
            embedding_weight=emb_weight,
            n_clusters=n_clusters,
            top_k_clusters=top_k,
            output_multiplier=model.output_multiplier,
        )
        hlm = hlm.to(device)
        hlm.build_clusters(verbose=verbose)
        hlm = hlm.to(dtype=next(model.parameters()).dtype)
        model._hlm = hlm
        info['hierarchical_lm'] = True
        info['hlm_n_clusters'] = n_clusters
        info['hlm_top_k_clusters'] = top_k
        if verbose:
            print(f"[Inference] Hierarchical LM Head: {n_clusters} clusters, "
                  f"top-{top_k} fine pass")

    info['model_dtype'] = next(model.parameters()).dtype
    return model, info


def _prepare_megakernel_bufs(model, device, verbose=True):
    """Pre-allocate mega-kernel buffers and pre-compute weights for eligible SSM layers."""
    fused_mod = _get_fused_ssm_layer()
    if fused_mod is None:
        if verbose:
            print("[Inference] Fused SSM layer kernels not available, skipping mega-kernel setup")
        return 0

    from .components.ssm import SelectiveSSMSSD

    n_prepared = 0
    for i, layer in enumerate(model.layers):
        block = layer.block
        if not isinstance(block, SelectiveSSMSSD):
            continue

        if not (hasattr(block.in_proj, 'w_val') and
                hasattr(block.x_proj, 'w_val') and
                hasattr(block.out_proj, 'w_val')):
            continue

        skip = False
        for proj_name in ('in_proj', 'x_proj', 'out_proj'):
            proj = getattr(block, proj_name)
            if (getattr(proj, 'correction_A', None) is not None or
                    getattr(proj, 'block_mixing', None) is not None or
                    getattr(proj, 'gate_post', None) is not None):
                if verbose:
                    print(f"[Inference] Layer {i} {proj_name} has extras, skipping mega-kernel")
                skip = True
                break
        if skip:
            continue

        block._mk_conv_weight = block.conv1d.weight.data.squeeze(1).contiguous().float()
        block._mk_conv_bias = block.conv1d.bias.data.contiguous().float()
        block._mk_dt_weight = block.dt_proj.weight.data.contiguous().float()
        block._mk_dt_bias = block.dt_proj.bias.data.contiguous().float()
        block._mk_A_log = block.A_log.data.contiguous().float()
        block._mk_D = block.D.data.contiguous().float()

        bufs = fused_mod.FusedSSMLayerBuffers.create(
            d_model=block.d_model,
            d_inner=block.d_inner,
            d_state=block.d_state,
            dt_rank=block.dt_rank,
            device=device,
        )
        block._megakernel_bufs = bufs
        n_prepared += 1

    if verbose:
        print(f"[Inference] Mega-kernel buffers prepared for {n_prepared} SSM layers")
    return n_prepared


def _precompute_entropy_screening(model, verbose=True):
    """Precompute L1 norms and sorted indices for entropy screening.
    Must be called BEFORE backend conversion (while raw weights accessible).
    """
    with torch.no_grad():
        if model.lm_head is not None and hasattr(model.lm_head, 'weight'):
            lm = model.lm_head
            if hasattr(lm, 'quantize_enabled') and lm.quantize_enabled:
                from .components.ternary import TernaryQuantize
                t_coeff = getattr(lm, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF)
                gs = getattr(lm, 'group_size', 128)
                W_q = TernaryQuantize.apply(lm.weight, gs, t_coeff)
                if lm.alpha is not None:
                    W_q = lm._apply_alpha(W_q)
            else:
                W_q = lm.weight
        elif model.lm_head is None:
            W_q = model.embedding.weight
        else:
            W_q = model.lm_head.weight

        row_l1 = W_q.abs().sum(dim=1)
        row_l1 = row_l1 * model.output_multiplier
        sorted_indices = row_l1.argsort(descending=True)

        model._es_row_l1 = row_l1
        model._es_sorted_indices = sorted_indices

        if verbose:
            V = row_l1.shape[0]
            print(f"[Inference] Entropy Screening: vocab={V}, "
                  f"L1 max={row_l1[sorted_indices[0]].item():.2f}, "
                  f"median={row_l1[sorted_indices[V // 2]].item():.2f}, "
                  f"min={row_l1[sorted_indices[-1]].item():.2f}")


def _entropy_screened_logits(x, model, k_screen=4096):
    """Compute logits using entropy screening - only evaluate top-K vocab by L1 bound.
    Returns (batch, 1, vocab_size) with -inf for screened entries.
    """
    sorted_indices = model._es_sorted_indices
    candidate_indices = sorted_indices[:k_screen]

    if hasattr(model.lm_head, 'w_val') if model.lm_head is not None else False:
        logits = model.lm_head(x)
        logits = logits * model.output_multiplier
        mask = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
        mask[candidate_indices] = True
        logits[:, ~mask] = float('-inf')
        return logits.unsqueeze(1)

    if model.lm_head is not None:
        W_full = model.lm_head.weight
        if hasattr(model.lm_head, 'quantize_enabled') and model.lm_head.quantize_enabled:
            from .components.ternary import TernaryQuantize
            t_coeff = getattr(model.lm_head, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF)
            gs = getattr(model.lm_head, 'group_size', 128)
            W_full = TernaryQuantize.apply(W_full, gs, t_coeff)
            if model.lm_head.alpha is not None:
                W_full = model.lm_head._apply_alpha(W_full)
    else:
        W_full = model.embedding.weight

    W_candidates = W_full[candidate_indices]
    candidate_logits = F.linear(x, W_candidates) * model.output_multiplier

    x_inf = x.abs().amax(dim=-1)
    champion_score = candidate_logits.max(dim=-1).values
    skip_upper_bound = model._es_row_l1[sorted_indices[k_screen]] * x_inf

    if (skip_upper_bound >= champion_score).any():
        if model.lm_head is not None:
            logits = model.lm_head(x) * model.output_multiplier
        else:
            logits = F.linear(x, model.embedding.weight) * model.output_multiplier
        return logits.unsqueeze(1)

    vocab_size = model._es_row_l1.shape[0]
    logits = torch.full((x.shape[0], vocab_size), float('-inf'), device=x.device, dtype=x.dtype)
    logits[:, candidate_indices] = candidate_logits
    return logits.unsqueeze(1)


@dataclass
class InferenceState:
    layer_states: List[Dict[str, Any]]
    v_first: Optional[torch.Tensor]
    position: int
    last_exit_layer: int = -1


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
            window_size = max(block.per_head_windows) if block.per_head_windows is not None else block.window_size
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


def clone_state(state: InferenceState) -> InferenceState:
    """Deep clone inference state so in-place mutations don't cross-contaminate."""
    cloned_layers = []
    for ls in state.layer_states:
        cloned = {}
        for k, v in ls.items():
            cloned[k] = v.clone() if isinstance(v, torch.Tensor) else v
        cloned_layers.append(cloned)
    return InferenceState(
        layer_states=cloned_layers,
        v_first=state.v_first.clone() if state.v_first is not None else None,
        position=state.position
    )


def _reset_inference_state(state: InferenceState):
    """Reset state tensors to zero in-place, preserving memory addresses for CUDA graph reuse."""
    state.position = 0
    state.last_exit_layer = -1
    for ls in state.layer_states:
        for k, v in ls.items():
            if isinstance(v, torch.Tensor):
                v.zero_()
            elif k == 'cache_len':
                ls[k] = 0
    if state.v_first is not None:
        state.v_first.zero_()


def ssm_ssd_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    h: torch.Tensor,
    conv_state: torch.Tensor
) -> torch.Tensor:
    """SSD single-step update (multi-head state space duality).

    Args:
        layer: ModelBlock with SelectiveSSMSSD
        x_t: (batch, d_model) - single token embedding
        h: (batch, n_heads, d_state, head_dim) - hidden state (mutated in-place)
        conv_state: (batch, d_inner, d_conv) - convolution state (mutated in-place)

    Returns:
        output: (batch, d_model)
    """
    block = layer.block
    batch_size = x_t.shape[0]

    x_norm = layer.norm(x_t)
    xz = block.in_proj(x_norm)
    x_inner, z = xz.chunk(2, dim=-1)

    # Update convolution state in-place: shift left, add new column on right
    x_inner_expanded = x_inner.unsqueeze(-1)  # (batch, d_inner, 1)
    conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
    conv_state[:, :, -1:] = x_inner_expanded
    x_conv = (conv_state * block.conv1d.weight.squeeze(1)).sum(dim=-1) + block.conv1d.bias
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
    fused_kernel = get_fused_ssm_kernel() if x_t.is_cuda else None
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
        h.copy_(h_new)
    else:
        # Compute dA: exp(dt_heads * A) where A is scalar per head
        # dt_heads: (batch, n_heads), A: (n_heads,)
        # Result: (batch, n_heads, 1)
        dA = torch.exp((dt_heads * A.unsqueeze(0)).unsqueeze(-1))  # (batch, n_heads, 1)

        # Compute dB * x: dt * B * x
        # dt_heads: (batch, n_heads)
        # B: (batch, d_state) -> need to broadcast to (batch, n_heads, d_state)
        # x_heads: (batch, n_heads, head_dim)
        B_expanded = B.unsqueeze(1).expand(batch_size, n_heads, -1).float()  # (batch, n_heads, d_state)

        # dB_x = dt_heads * (B expanded) * x_heads
        # (batch, n_heads, 1, 1) * (batch, n_heads, d_state, 1) * (batch, n_heads, 1, head_dim)
        dB_x = (dt_heads.unsqueeze(-1).unsqueeze(-1) * B_expanded.unsqueeze(-1) *
                x_heads.unsqueeze(2))  # (batch, n_heads, d_state, head_dim)

        # Update state in-place: h = dA * h + dB_x
        # h: (batch, n_heads, d_state, head_dim)
        # dA: (batch, n_heads, 1)
        # dB_x: (batch, n_heads, d_state, head_dim)
        h.mul_(dA.unsqueeze(-1).float()).add_(dB_x.float())

        # Compute output: y = h @ C
        # C: (batch, d_state) -> (batch, 1, 1, d_state) for broadcasting
        # h: (batch, n_heads, d_state, head_dim)
        # Result should be (batch, n_heads, head_dim)
        C_expanded = C.unsqueeze(1).expand(batch_size, n_heads, -1).float()  # (batch, n_heads, d_state)
        y = (h * C_expanded.unsqueeze(-1)).sum(dim=2)  # (batch, n_heads, head_dim)

        # Reshape y back to (batch, d_inner)
        y = y.reshape(batch_size, block.d_inner)

        # Add residual
        y = y + x_conv * block.D  # (batch, d_inner)

    # Gate and project
    silu_z = F.silu(z)
    y = silu_z * y.to(x_t.dtype)

    output = x_t + block.out_proj(y)

    return output


def _ssm_ssd_megakernel(layer, x_t, h, conv_state, bufs, fused_mod):
    """SSM SSD step via fused mega-kernels (K1+K2a+K2b+K3+K4 = 5 launches).

    Batch dimension squeezed for all tensors (batch=1 only).
    h and conv_state mutated in-place via tensor views.
    """
    block = layer.block
    d_inner = block.d_inner
    d_model = block.d_model

    x_1d = x_t.squeeze(0)
    h_3d = h.squeeze(0)
    conv_2d = conv_state.squeeze(0)

    # K1: RMSNorm + in_proj ternary GEMV
    fused_mod.fused_rmsnorm_inproj_step(
        x=x_1d,
        norm_weight=layer.norm.weight,
        w_val=block.in_proj.w_val,
        w_sign=block.in_proj.w_sign,
        scales=block.in_proj.scales,
        y_buf=bufs.buf_xz,
        d_model=d_model,
        group_size=block.in_proj.group_size,
    )

    x_inner = bufs.buf_xz[:d_inner]
    z = bufs.buf_xz[d_inner:]

    # K2: conv1d update + SiLU + x_proj GEMV
    fused_mod.fused_conv_silu_xproj_step(
        x_inner=x_inner,
        conv_state=conv_2d,
        conv_weight=block._mk_conv_weight,
        conv_bias=block._mk_conv_bias,
        xproj_w_val=block.x_proj.w_val,
        xproj_w_sign=block.x_proj.w_sign,
        xproj_scales=block.x_proj.scales,
        x_conv_buf=bufs.buf_xconv,
        x_dbc_buf=bufs.buf_xdbc,
        d_inner=d_inner,
        d_conv=block.d_conv,
        group_size=block.x_proj.group_size,
    )

    # K3: dt_proj + softplus + SSM recurrence
    fused_mod.fused_dt_ssm_step(
        x_dbc=bufs.buf_xdbc,
        x_conv=bufs.buf_xconv,
        h=h_3d,
        dt_proj_weight=block._mk_dt_weight,
        dt_proj_bias=block._mk_dt_bias,
        A_log=block._mk_A_log,
        D=block._mk_D,
        y_buf=bufs.buf_y,
    )

    # K4: SiLU gating + out_proj ternary GEMV + residual
    fused_mod.fused_gate_outproj_residual_step(
        y=bufs.buf_y,
        z=z,
        x_t=x_1d,
        w_val=block.out_proj.w_val,
        w_sign=block.out_proj.w_sign,
        scales=block.out_proj.scales,
        output_buf=bufs.buf_output,
        d_model=d_model,
        group_size=block.out_proj.group_size,
    )

    return bufs.buf_output.unsqueeze(0)


def ssm_ssd_step_fused(layer, x_t, h, conv_state):
    """SSD step with fused kernels. Fallback chain:
       mega-kernel (5 launches) -> Triton fused recurrence (~15) -> PyTorch (~25-30)."""
    # Tier 1: Mega-kernel path (K1+K2a+K2b+K3+K4)
    block = layer.block
    bufs = getattr(block, '_megakernel_bufs', None)
    if bufs is not None and x_t.is_cuda:
        fused_mod = _get_fused_ssm_layer()
        if fused_mod is not None:
            try:
                return _ssm_ssd_megakernel(layer, x_t, h, conv_state, bufs, fused_mod)
            except Exception as e:
                if not torch.compiler.is_compiling():
                    print(f"[Inference] Mega-kernel failed, falling back: {e}")
                block._megakernel_bufs = None

    # Tier 2: Triton fused recurrence (only SSM recurrence is fused)
    triton_ssm = _get_triton_ssm() if x_t.is_cuda else None
    if triton_ssm is None:
        return ssm_ssd_step(layer, x_t, h, conv_state)

    batch_size = x_t.shape[0]

    x_norm = layer.norm(x_t)
    xz = block.in_proj(x_norm)
    x_inner, z = xz.chunk(2, dim=-1)

    x_inner_expanded = x_inner.unsqueeze(-1)
    conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
    conv_state[:, :, -1:] = x_inner_expanded
    x_conv = (conv_state * block.conv1d.weight.squeeze(1)).sum(dim=-1) + block.conv1d.bias
    x_conv = F.silu(x_conv)

    x_dbc = block.x_proj(x_conv)
    dt_input = x_dbc[:, :block.dt_rank]
    B = x_dbc[:, block.dt_rank:block.dt_rank + block.d_state]
    C = x_dbc[:, block.dt_rank + block.d_state:]

    dt = F.softplus(block.dt_proj(dt_input))
    A = -torch.exp(block.A_log)

    n_heads = block.n_heads
    head_dim = block.d_inner // n_heads

    dt_heads = dt.view(batch_size, n_heads, head_dim).mean(dim=-1).float()
    x_heads = x_conv.view(batch_size, n_heads, head_dim).float()

    y = triton_ssm(h, dt_heads, A.float(), B.float(), C.float(), x_heads, block.D.float())

    silu_z = F.silu(z)
    y = silu_z * y.to(x_t.dtype)
    output = x_t + block.out_proj(y)

    return output


def ssm_mamba1_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    h: torch.Tensor,
    conv_state: torch.Tensor
) -> torch.Tensor:
    block = layer.block
    batch_size = x_t.shape[0]

    x_norm = layer.norm(x_t)
    xz = block.in_proj(x_norm)
    x_inner, z = xz.chunk(2, dim=-1)

    x_inner_expanded = x_inner.unsqueeze(-1)
    conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
    conv_state[:, :, -1:] = x_inner_expanded
    x_conv = (conv_state * block.conv1d.weight.squeeze(1)).sum(dim=-1) + block.conv1d.bias
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

    h.mul_(dA).add_(dB_x)
    y = (h * C.unsqueeze(1).to(torch.float32)).sum(dim=-1).to(x_t.dtype) + x_conv * block.D

    silu_z = F.silu(z)
    y = silu_z * y

    output = x_t + block.out_proj(y)

    return output


def swa_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: int,
    position: int
) -> Tuple[torch.Tensor, int]:
    block = layer.block
    batch_size = x_t.shape[0]

    x_norm = block.attn_norm(x_t)

    q = block.q_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)
    k = block.k_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)
    v = block.v_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)

    rope_cache_size = block.rotary.cos_cached.shape[2]
    pos_wrapped = position % rope_cache_size
    cos_cached = block.rotary.cos_cached[:, :, pos_wrapped, :]
    sin_cached = block.rotary.sin_cached[:, :, pos_wrapped, :]

    def rotate_half(x_val):
        x_chunked = x_val.chunk(2, dim=-1)
        return torch.cat([-x_chunked[1], x_chunked[0]], dim=-1)

    q = q * cos_cached + rotate_half(q) * sin_cached
    k = k * cos_cached + rotate_half(k) * sin_cached

    window_size = max(block.per_head_windows) if block.per_head_windows is not None else block.window_size
    cache_idx = cache_len % window_size
    k_cache[:, :, cache_idx, :] = k
    v_cache[:, :, cache_idx, :] = v

    # TODO: Sink tokens in inference: store first n_sink KVs separately in state,
    # concatenate during attention. Currently working in training via mask. Inference
    # optimization deferred as sink tokens primarily benefit long sequences (>window_size).

    cache_len_new = cache_len + 1
    valid_len = min(cache_len_new, window_size)

    scale = 1.0 / math.sqrt(block.head_dim)
    scores = torch.matmul(q.unsqueeze(-2), k_cache[:, :, :valid_len, :].transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)

    attn_out = torch.matmul(attn_weights, v_cache[:, :, :valid_len, :])
    attn_out = attn_out.squeeze(-2).view(batch_size, -1)

    if getattr(block, 'use_head_gating', False):
        gate = torch.sigmoid(block.head_gate_proj(x_norm) + block.head_gate_bias)
        gate = gate.unsqueeze(-1)
        attn_out_shaped = attn_out.view(batch_size, block.n_heads, -1)
        attn_out_shaped = attn_out_shaped * gate
        attn_out = attn_out_shaped.view(batch_size, -1)

    attn_out = block.out_proj(attn_out)

    x = x_t + attn_out

    x_norm = block.mlp_norm(x)
    gate_out = F.silu(block.mlp_gate(x_norm)) * block.mlp_up(x_norm)
    mlp_out = block.mlp_down(gate_out)

    output = x + mlp_out

    return output, cache_len_new


def rotate_half(x):
    x_chunked = x.chunk(2, dim=-1)
    return torch.cat([-x_chunked[1], x_chunked[0]], dim=-1)


def swa_step_graphable(
    layer,
    x_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_rope: torch.Tensor,
    sin_rope: torch.Tensor,
    cache_write_idx: torch.Tensor,
) -> torch.Tensor:
    """SWA single-step inference for CUDA graph capture.
    
    Args:
        layer: ModelBlock with SlidingWindowAttention
        x_t: (batch, d_model) - single token embedding
        k_cache: (batch, n_heads, window_size, head_dim) - key cache
        v_cache: (batch, n_heads, window_size, head_dim) - value cache
        cos_rope: (batch, n_heads, head_dim) - pre-computed RoPE cosine
        sin_rope: (batch, n_heads, head_dim) - pre-computed RoPE sine
        cache_write_idx: (1,) - write position in circular cache
    """
    block = layer.block
    batch_size = x_t.shape[0]

    x_norm = block.attn_norm(x_t)

    q = block.q_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)
    k = block.k_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)
    v = block.v_proj(x_norm).view(batch_size, block.n_heads, block.head_dim)

    # Apply pre-computed RoPE (computed outside graph to support dynamic position)
    q = q * cos_rope + rotate_half(q) * sin_rope
    k = k * cos_rope + rotate_half(k) * sin_rope

    # Update cache using scatter compatible with CUDA graphs
    idx = cache_write_idx.view(1, 1, 1, 1).expand(batch_size, block.n_heads, 1, block.head_dim)
    k_cache.scatter_(2, idx, k.unsqueeze(2))
    v_cache.scatter_(2, idx, v.unsqueeze(2))

    # Compute attention over full cache window
    scale = 1.0 / math.sqrt(block.head_dim)
    scores = torch.matmul(q.unsqueeze(-2), k_cache.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)

    attn_out = torch.matmul(attn_weights, v_cache)
    attn_out = attn_out.squeeze(-2).view(batch_size, -1)

    if getattr(block, 'use_head_gating', False):
        gate = torch.sigmoid(block.head_gate_proj(x_norm) + block.head_gate_bias)
        gate = gate.unsqueeze(-1)
        attn_out_shaped = attn_out.view(batch_size, block.n_heads, -1)
        attn_out_shaped = attn_out_shaped * gate
        attn_out = attn_out_shaped.view(batch_size, -1)

    attn_out = block.out_proj(attn_out)

    x = x_t + attn_out

    x_norm = block.mlp_norm(x)
    gate_out = F.silu(block.mlp_gate(x_norm)) * block.mlp_up(x_norm)
    mlp_out = block.mlp_down(gate_out)

    output = x + mlp_out

    return output



def rwkv7_step(
    layer: 'ModelBlock',
    x_t: torch.Tensor,
    state: torch.Tensor,
    x_prev: torch.Tensor,
    v_first: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    block = layer.block
    batch_size = x_t.shape[0]
    d_model = block.d_model

    x_norm = layer.norm(x_t)

    xx = x_prev - x_norm

    xr = x_norm + xx * block.x_r
    xk = x_norm + xx * block.x_k
    xv = x_norm + xx * block.x_v
    xw = x_norm + xx * block.x_w
    xa = x_norm + xx * block.x_a
    xg = x_norm + xx * block.x_g

    r = block.W_r(xr).view(batch_size, block.n_heads, block.head_size)
    k = block.W_k(xk).view(batch_size, block.n_heads, block.head_size)
    v = block.W_v(xv).view(batch_size, block.n_heads, block.head_size)

    if v_first is None:
        v_first_new = v.detach()
    else:
        v_first_shaped = v_first.view(batch_size, block.n_heads, block.head_size)
        alpha = torch.sigmoid(block.v0 + (xv @ block.v1) @ block.v2).view(batch_size, block.n_heads, block.head_size)
        v = v + (v_first_shaped - v) * alpha
        v_first_new = v.detach()
        v_first.copy_(v_first_new)

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

    state_aa_bb = torch.matmul(state, aa_bb)
    state.mul_(w_mat).add_(state_aa_bb).add_(vk)

    out = (state @ r_mat).squeeze(-1)
    out = out.view(batch_size, -1)

    out = block.group_norm(out.unsqueeze(1)).squeeze(1)

    r_flat = r.view(batch_size, -1)
    k_flat = k_scaled.view(batch_size, -1)
    v_flat = v.view(batch_size, -1)
    r_k = block.r_k.view(1, block.n_heads, block.head_size).expand_as(r).reshape(batch_size, -1)
    la_term = (r_flat * k_flat * r_k).sum(dim=-1, keepdim=True) * v_flat
    out = out + la_term

    output = x_t + block.W_o(out * g.view(batch_size, -1))

    x_prev.copy_(x_norm)
    return output, v_first


def model_step_graphable(
    model: ModelV2,
    token_buf: torch.Tensor,
    state: InferenceState,
    rope_cos_sin: List[Tuple[torch.Tensor, torch.Tensor]],
    cache_write_indices: List[torch.Tensor]
) -> torch.Tensor:
    batch_size = token_buf.shape[0]

    x = model.embedding(token_buf)
    x = x.squeeze(1) if x.dim() > 2 else x

    swa_counter = 0
    for i, layer in enumerate(model.layers):
        block = layer.block
        layer_state = state.layer_states[i]

        if isinstance(block, SelectiveSSMSSD):
            h = layer_state['h']
            conv_state = layer_state['conv_state']
            x = ssm_ssd_step_fused(layer, x, h, conv_state)

        elif isinstance(block, SelectiveSSM):
            h = layer_state['h']
            conv_state = layer_state['conv_state']
            x = ssm_mamba1_step(layer, x, h, conv_state)

        elif isinstance(block, SlidingWindowAttention):
            k_cache = layer_state['k_cache']
            v_cache = layer_state['v_cache']

            cos_rope, sin_rope = rope_cos_sin[swa_counter]
            x = swa_step_graphable(
                layer, x, k_cache, v_cache,
                cos_rope, sin_rope, cache_write_indices[swa_counter]
            )
            swa_counter += 1

        elif getattr(layer, 'is_rwkv7', False):
            state_mat = layer_state['state']
            x_prev = layer_state['x_prev']

            x, state.v_first = rwkv7_step(
                layer, x, state_mat, x_prev, state.v_first
            )

    x = model.final_norm(x)
    if hasattr(model, '_hlm') and model._hlm is not None:
        return model._hlm(x)
    if model.lm_head is not None:
        logits = model.lm_head(x)
    else:
        logits = F.linear(x, model.embedding.weight)
    logits = logits * model.output_multiplier
    if hasattr(model, '_es_row_l1') and model._es_row_l1 is not None:
        k = getattr(model, '_es_k_screen', 4096)
        mask = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
        mask[model._es_sorted_indices[:k]] = True
        logits = logits.masked_fill(~mask, float('-inf'))
    return logits.unsqueeze(1)


def model_step(
    model: ModelV2,
    token_id: torch.Tensor,
    state: InferenceState,
    max_layers: Optional[int] = None
) -> Tuple[torch.Tensor, InferenceState]:
    batch_size = token_id.shape[0]
    device = token_id.device

    x = model.embedding(token_id)
    x = x.squeeze(1) if x.dim() > 2 else x

    # Early exit config
    ee_enabled = getattr(model.config, 'use_early_exit', False) and not model.training
    ee_interval = getattr(model.config, 'early_exit_interval', 4)
    ee_threshold = getattr(model.config, 'early_exit_threshold', 0.99)
    x_prev_exit = None
    state.last_exit_layer = len(model.layers)

    for i, layer in enumerate(model.layers):
        if max_layers is not None and i >= max_layers:
            break
        block = layer.block
        layer_state = state.layer_states[i]

        # Save hidden state at start of each group for early exit
        if ee_enabled and i % ee_interval == 0:
            x_prev_exit = x.detach()

        if isinstance(block, SelectiveSSMSSD):
            h = layer_state['h']
            conv_state = layer_state['conv_state']
            x = ssm_ssd_step(layer, x, h, conv_state)

        elif isinstance(block, SelectiveSSM):
            h = layer_state['h']
            conv_state = layer_state['conv_state']
            x = ssm_mamba1_step(layer, x, h, conv_state)

        elif isinstance(block, SlidingWindowAttention):
            k_cache = layer_state['k_cache']
            v_cache = layer_state['v_cache']
            cache_len = layer_state['cache_len']

            x, cache_len_new = swa_step(
                layer, x, k_cache, v_cache, cache_len, state.position
            )
            layer_state['cache_len'] = cache_len_new

        elif getattr(layer, 'is_rwkv7', False):
            state_mat = layer_state['state']
            x_prev = layer_state['x_prev']

            x, state.v_first = rwkv7_step(
                layer, x, state_mat, x_prev, state.v_first
            )

        # Early exit check at end of each group
        if (ee_enabled and x_prev_exit is not None
                and (i + 1) % ee_interval == 0
                and (i + 1) >= 2 * ee_interval):
            cos_sim = F.cosine_similarity(
                x.float().view(batch_size, -1),
                x_prev_exit.float().view(batch_size, -1),
                dim=-1
            ).mean()
            if cos_sim.item() > ee_threshold:
                state.last_exit_layer = i + 1
                break

    x = model.final_norm(x)

    if hasattr(model, '_hlm') and model._hlm is not None:
        logits = model._hlm(x)
    elif hasattr(model, '_es_row_l1') and model._es_row_l1 is not None:
        logits = _entropy_screened_logits(x, model, getattr(model, '_es_k_screen', 4096))
    else:
        if model.lm_head is not None:
            logits = model.lm_head(x)
        else:
            logits = F.linear(x, model.embedding.weight)
        logits = logits * model.output_multiplier
        logits = logits.unsqueeze(1)

    state.position += 1

    return logits, state


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


class CUDAGraphRunner:
    def __init__(self, model: ModelV2, state: InferenceState, device='cuda'):
        self.model = model
        self.state = state
        self.device = device

        self.swa_layer_indices = []
        self.swa_window_sizes = []
        self.swa_rotary_modules = []
        for i, layer in enumerate(model.layers):
            if isinstance(layer.block, SlidingWindowAttention):
                self.swa_layer_indices.append(i)
                window_size = max(layer.block.per_head_windows) if layer.block.per_head_windows is not None else layer.block.window_size
                self.swa_window_sizes.append(window_size)
                self.swa_rotary_modules.append(layer.block.rotary)

        self.token_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.cache_write_indices = []
        for idx, ws in zip(self.swa_layer_indices, self.swa_window_sizes):
            cache_len = state.layer_states[idx]['cache_len']
            self.cache_write_indices.append(
                torch.tensor([cache_len % ws], dtype=torch.long, device=device)
            )

        # Pre-allocate rope cos/sin tensors (batch=1, n_heads, head_dim)
        self.rope_cos_sin = []
        for i, layer in enumerate(model.layers):
            if isinstance(layer.block, SlidingWindowAttention):
                batch_size = 1
                n_heads = layer.block.n_heads
                head_dim = layer.block.head_dim
                cos_buf = torch.zeros(batch_size, n_heads, head_dim, device=device, dtype=model.parameters().__next__().dtype)
                sin_buf = torch.zeros(batch_size, n_heads, head_dim, device=device, dtype=model.parameters().__next__().dtype)
                self.rope_cos_sin.append((cos_buf, sin_buf))

        self.graph = None
        self.logits_static = None
        self._captured = False

    def _compute_rope_cos_sin(self):
        """Compute RoPE cos/sin for current position and update buffers."""
        position = self.state.position
        for i, (rotary, (cos_buf, sin_buf)) in enumerate(zip(self.swa_rotary_modules, self.rope_cos_sin)):
            # Get cos/sin for this position from the rotary cache
            # Use modulo to wrap around if position exceeds cache size
            rope_cache_size = rotary.cos_cached.shape[2]
            pos_wrapped = position % rope_cache_size
            cos_cached = rotary.cos_cached[:, :, pos_wrapped, :]
            sin_cached = rotary.sin_cached[:, :, pos_wrapped, :]
            cos_buf.copy_(cos_cached)
            sin_buf.copy_(sin_cached)

    def _update_cache_write_indices(self):
        """Update cache write indices based on current state."""
        for i, (swa_idx, ws) in enumerate(zip(self.swa_layer_indices, self.swa_window_sizes)):
            cache_len = self.state.layer_states[swa_idx]['cache_len']
            self.cache_write_indices[i].fill_(cache_len % ws)

    def _advance_state(self):
        """Advance position and cache_len counters."""
        self.state.position += 1
        for swa_idx in self.swa_layer_indices:
            self.state.layer_states[swa_idx]['cache_len'] += 1

    def warmup(self, n_steps: int = 3):
        print(f"[CUDAGraph] Warming up ({n_steps} steps)...")
        for i in range(n_steps):
            self.token_buf.fill_(0)
            self._compute_rope_cos_sin()
            self._update_cache_write_indices()
            with torch.no_grad():
                logits = model_step_graphable(
                    self.model, self.token_buf, self.state,
                    self.rope_cos_sin, self.cache_write_indices
                )
            self._advance_state()
        print(f"[CUDAGraph] Warmup complete. Position={self.state.position}")

    def capture(self):
        print("[CUDAGraph] Capturing graph...")
        self.token_buf.fill_(0)
        self._compute_rope_cos_sin()
        self._update_cache_write_indices()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits_static = model_step_graphable(
                self.model, self.token_buf, self.state,
                self.rope_cos_sin, self.cache_write_indices
            )

        self._advance_state()
        self._captured = True
        print(f"[CUDAGraph] Graph captured! Position={self.state.position}")

    def step(self, token_id: int) -> torch.Tensor:
        assert self._captured, "Must call capture() first"
        self.token_buf[0, 0] = token_id
        self._compute_rope_cos_sin()
        self._update_cache_write_indices()
        self.graph.replay()
        self._advance_state()
        return self.logits_static

@torch.no_grad()
def generate_with_cuda_graph(
    model: ModelV2,
    prompt_ids: torch.Tensor,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    warmup_steps: int = 3,
    verbose: bool = False
) -> List[int]:
    device = next(model.parameters()).device
    batch_size = 1

    # Check for cached CUDA graph runner
    cached_runner = getattr(model, '_cuda_graph_runner', None)

    if cached_runner is not None:
        # Reuse cached runner's state (same tensor memory as captured graph)
        state = cached_runner.state
        _reset_inference_state(state)
    else:
        state = init_state(model, batch_size, device)

    seq = []
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    logits = None
    for i in range(prompt_ids.shape[1]):
        token = prompt_ids[:, i:i+1].to(device)
        logits, state = model_step(model, token, state)

    if verbose:
        print(f"[CUDAGraph] Prefill complete. Position={state.position}")

    for i, layer in enumerate(model.layers):
        if isinstance(layer.block, SlidingWindowAttention):
            cache_len = state.layer_states[i]['cache_len']
            ws = max(layer.block.per_head_windows) if layer.block.per_head_windows is not None else layer.block.window_size
            if cache_len < ws:
                extra = ws - cache_len
                if verbose:
                    print(f"[CUDAGraph] Saturating SWA cache at layer {i}: {extra} extra steps needed")
                for _ in range(extra):
                    token_id = sample_token(logits, temperature, top_k, top_p)
                    seq.append(token_id)
                    token = torch.tensor([[token_id]], device=device, dtype=torch.long)
                    logits, state = model_step(model, token, state)

    if cached_runner is None:
        runner = CUDAGraphRunner(model, state, device)
        runner.warmup(warmup_steps)
        runner.capture()
        model._cuda_graph_runner = runner
        if verbose:
            print("[CUDAGraph] Runner cached on model for reuse")
    else:
        runner = cached_runner
        runner._update_cache_write_indices()
        if verbose:
            print(f"[CUDAGraph] Reusing cached runner. Position={state.position}")

    last_token = sample_token(logits, temperature, top_k, top_p)
    seq.append(last_token)

    for _ in range(max_tokens - len(seq)):
        graph_logits = runner.step(last_token)
        next_token = sample_token(graph_logits, temperature, top_k, top_p)
        seq.append(next_token)
        last_token = next_token

    return seq


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
