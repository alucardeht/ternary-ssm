import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple
from .inference import (
    InferenceState, init_state, clone_state, model_step, sample_token
)
from .components.ternary import DEFAULT_THRESHOLD_COEFF


@torch.no_grad()
def stochastic_ternary_quantize(
    weight: torch.Tensor,
    group_size: int = 128,
    threshold_coeff: float = DEFAULT_THRESHOLD_COEFF,
    temperature: float = 0.3
) -> torch.Tensor:
    """Stochastic ternary quantization with probabilistic zero/nonzero sampling.

    For each weight group: compute the probability that magnitude exceeds threshold,
    then sample is_nonzero from Bernoulli distribution. Results in stochastic
    patterns suitable for ensemble inference diversity.

    Args:
        weight: Weight tensor to quantize
        group_size: Group size for per-group normalization
        threshold_coeff: Threshold coefficient (threshold = coeff * mean(|normalized|))
        temperature: Temperature for sigmoid slope (lower = sharper)

    Returns:
        Stochastically quantized weight tensor (same shape/dtype as input)
    """
    orig_shape = weight.shape
    flat = weight.reshape(-1)

    pad_len = (group_size - flat.numel() % group_size) % group_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))
    grouped = flat.reshape(-1, group_size)

    scales = grouped.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = grouped / scales

    abs_normalized = normalized.abs()
    threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
    distance = abs_normalized - threshold

    dist_scale = distance.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    p_nonzero = torch.sigmoid(distance / (temperature * dist_scale))
    is_nonzero = torch.bernoulli(p_nonzero)

    quantized = normalized.sign() * is_nonzero

    result = (quantized * scales).reshape(-1)
    if pad_len > 0:
        result = result[:weight.numel()]

    return result.reshape(orig_shape)


def inject_stochastic_weights(
    model,
    temperature: float = 0.3,
    seed: Optional[int] = None
) -> Dict:
    """Inject stochastic ternary quantization into all TernaryLinear layers.

    Saves original state (ste_type, _cached_qweight, _cache_valid) before injection.
    Replaces cached weights with stochastically quantized versions.

    Args:
        model: ModelV2 instance
        temperature: Temperature for stochastic quantization
        seed: Random seed (applied to both torch and cuda if provided)

    Returns:
        Dictionary mapping module id -> original state for restoration
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    restore_info = {}

    for name, module in model.named_modules():
        if not (hasattr(module, 'quantize_enabled') and
                hasattr(module, 'group_size') and
                hasattr(module, 'weight')):
            continue

        module_id = id(module)
        restore_info[module_id] = {
            'name': name,
            'ste_type': getattr(module, 'ste_type', 'standard'),
            '_cached_qweight': getattr(module, '_cached_qweight', None),
            '_cache_valid': getattr(module, '_cache_valid', False),
        }

        stochastic_q = stochastic_ternary_quantize(
            module.weight.data,
            group_size=module.group_size,
            threshold_coeff=getattr(module, 'threshold_coeff', DEFAULT_THRESHOLD_COEFF),
            temperature=temperature
        )

        module._cached_qweight = stochastic_q
        module._cache_valid = True
        module.ste_type = 'standard'

    return restore_info


def restore_weights(model, restore_info: Dict) -> None:
    """Restore original weight quantization state before STEWS injection.

    Args:
        model: ModelV2 instance
        restore_info: Dictionary returned from inject_stochastic_weights
    """
    for name, module in model.named_modules():
        module_id = id(module)
        if module_id not in restore_info:
            continue

        info = restore_info[module_id]
        module.ste_type = info['ste_type']
        module._cached_qweight = info['_cached_qweight']
        module._cache_valid = info['_cache_valid']


@torch.no_grad()
def generate_stews(
    model,
    prompt_ids: List[int],
    max_tokens: int = 100,
    K: int = 5,
    sampling_temperature: float = 1.0,
    stews_temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    verbose: bool = False
) -> List[int]:
    """Generate tokens using STEWS (Stochastic Ternary Weight Ensemble Sampling).

    Ensemble approach: K-1 stochastic passes + 1 deterministic pass per position.
    All K passes use same inference state (cloned), then logits are averaged.

    Args:
        model: ModelV2 instance
        prompt_ids: Initial token sequence to prefill
        max_tokens: Number of tokens to generate
        K: Ensemble size (number of stochastic passes per position)
        sampling_temperature: Temperature for final sampling
        stews_temperature: Temperature for stochastic quantization
        top_k: top-k sampling constraint
        top_p: nucleus sampling constraint
        verbose: Print progress every 10 tokens

    Returns:
        List of generated token IDs
    """
    model.eval()
    device = next(model.parameters()).device

    start_time = time.time()

    state = init_state(model, batch_size=1, device=device)

    for token_id in prompt_ids:
        token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
        _, state = model_step(model, token_tensor, state)

    generated = []

    for gen_step in range(max_tokens):
        all_logits = []

        for k in range(K):
            if k < K - 1:
                state_clone = clone_state(state)
                restore_info = inject_stochastic_weights(model, temperature=stews_temperature)

                logits_k, _ = model_step(model, torch.tensor([[0]], device=device, dtype=torch.long), state_clone)
                logits_k = logits_k.squeeze(0).squeeze(0)

                restore_weights(model, restore_info)
                all_logits.append(logits_k)
            else:
                logits_k, state = model_step(model, torch.tensor([[0]], device=device, dtype=torch.long), state)
                all_logits.append(logits_k.squeeze(0).squeeze(0))

        avg_logits = torch.stack(all_logits).mean(0)
        token_id = sample_token(avg_logits.unsqueeze(0).unsqueeze(0),
                               temperature=sampling_temperature, top_k=top_k, top_p=top_p)
        generated.append(token_id)

        if verbose and (gen_step + 1) % 10 == 0:
            print(f"[STEWS] Generated {gen_step + 1}/{max_tokens} tokens")

    elapsed = time.time() - start_time
    print(f"[STEWS] Generation complete: {max_tokens} tokens in {elapsed:.2f}s ({max_tokens/elapsed:.1f} tok/s)")

    return generated


@torch.no_grad()
def evaluate_stews_ppl(
    model,
    token_ids: List[int],
    K: int = 5,
    stews_temperature: float = 0.3,
    seq_len: int = 256,
    stride: int = 128,
    verbose: bool = False,
    max_batches: Optional[int] = None
) -> dict:
    """Evaluate model PPL using STEWS ensemble.

    Sliding window evaluation: for each position, use K-ensemble to predict next token,
    compute cross-entropy loss, accumulate NLL.

    Args:
        model: ModelV2 instance
        token_ids: Token sequence to evaluate
        K: Ensemble size
        stews_temperature: Temperature for stochastic quantization
        seq_len: Context window length
        stride: Stride for sliding window
        verbose: Print progress
        max_batches: Maximum windows to evaluate (None = all)

    Returns:
        Dict with 'ppl', 'avg_nll', 'n_tokens', 'time_sec'
    """
    model.eval()
    device = next(model.parameters()).device

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, device=device, dtype=torch.long)
    else:
        token_ids = token_ids.to(device)

    start_time = time.time()
    total_nll = 0.0
    n_tokens = 0
    window_count = 0

    for i in range(0, len(token_ids) - 1, stride):
        if max_batches is not None and window_count >= max_batches:
            break

        context = token_ids[i:min(i + seq_len, len(token_ids))]
        target_idx = min(i + seq_len, len(token_ids) - 1)
        target_token = token_ids[target_idx].item()

        state = init_state(model, batch_size=1, device=device)

        for j in range(len(context) - 1):
            token_tensor = context[j:j+1].unsqueeze(0)
            _, state = model_step(model, token_tensor, state)

        all_logits = []
        for k in range(K):
            if k < K - 1:
                state_clone = clone_state(state)
                restore_info = inject_stochastic_weights(model, temperature=stews_temperature)
                logits_k, _ = model_step(model, context[-1:].unsqueeze(0), state_clone)
                logits_k = logits_k.squeeze(0).squeeze(0)
                restore_weights(model, restore_info)
                all_logits.append(logits_k)
            else:
                logits_k, _ = model_step(model, context[-1:].unsqueeze(0), state)
                all_logits.append(logits_k.squeeze(0).squeeze(0))

        avg_logits = torch.stack(all_logits).mean(0)
        log_probs = F.log_softmax(avg_logits, dim=-1)
        nll = -log_probs[target_token].item()

        total_nll += nll
        n_tokens += 1
        window_count += 1

        if verbose and window_count % 10 == 0:
            print(f"[STEWS PPL] Processed {window_count} windows, avg_nll={total_nll/n_tokens:.4f}")

    elapsed = time.time() - start_time
    avg_nll = total_nll / n_tokens if n_tokens > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_nll)).item()

    return {
        'ppl': ppl,
        'avg_nll': avg_nll,
        'n_tokens': n_tokens,
        'time_sec': elapsed
    }


@torch.no_grad()
def compare_stews_vs_deterministic(
    model,
    token_ids: List[int],
    K_values: List[int] = [1, 3, 5],
    stews_temperature: float = 0.3,
    max_positions: int = 200
) -> dict:
    """Compare STEWS with different ensemble sizes against deterministic baseline (K=1).

    Runs evaluate_stews_ppl for K=1 (deterministic) and each K in K_values.
    Prints comparison table and returns results.

    Args:
        model: ModelV2 instance
        token_ids: Token sequence to evaluate
        K_values: List of ensemble sizes to test
        stews_temperature: Temperature for stochastic quantization
        max_positions: Limit evaluation windows (for faster benchmarking)

    Returns:
        Dict mapping K -> {'ppl': float, 'avg_nll': float, 'n_tokens': int, 'time_sec': float}
    """
    print("\n[STEWS Comparison] Starting evaluation...")
    print(f"Token sequence length: {len(token_ids)}")
    print(f"K values: {K_values}")
    print(f"STEWS temperature: {stews_temperature}\n")

    model.eval()

    results = {}

    baseline_K = 1
    print(f"[STEWS] Running baseline (K={baseline_K}, deterministic)...")
    baseline_result = evaluate_stews_ppl(
        model, token_ids, K=baseline_K,
        stews_temperature=stews_temperature,
        seq_len=256, stride=128,
        verbose=False,
        max_batches=max_positions // 128
    )
    results[baseline_K] = baseline_result
    print(f"  Baseline K={baseline_K}: PPL={baseline_result['ppl']:.2f}, "
          f"avg_nll={baseline_result['avg_nll']:.4f}, time={baseline_result['time_sec']:.2f}s\n")

    for K in K_values:
        if K == baseline_K:
            continue
        print(f"[STEWS] Running ensemble (K={K})...")
        result = evaluate_stews_ppl(
            model, token_ids, K=K,
            stews_temperature=stews_temperature,
            seq_len=256, stride=128,
            verbose=False,
            max_batches=max_positions // 128
        )
        results[K] = result
        ppl_delta = result['ppl'] - baseline_result['ppl']
        ppl_pct = (ppl_delta / baseline_result['ppl']) * 100
        print(f"  K={K}: PPL={result['ppl']:.2f} (Δ{ppl_pct:+.1f}%), "
              f"avg_nll={result['avg_nll']:.4f}, time={result['time_sec']:.2f}s\n")

    print("\n[STEWS Comparison] Results Table:")
    print("K\tPPL\t\tΔ%\t\tAvg_NLL\t\tTokens\t\tTime(s)")
    print("-" * 80)
    for K in sorted(results.keys()):
        res = results[K]
        if K == baseline_K:
            delta_str = "baseline"
        else:
            delta = ((res['ppl'] - baseline_result['ppl']) / baseline_result['ppl']) * 100
            delta_str = f"{delta:+.1f}%"
        print(f"{K}\t{res['ppl']:.2f}\t\t{delta_str}\t\t{res['avg_nll']:.4f}\t\t"
              f"{res['n_tokens']}\t\t{res['time_sec']:.2f}")

    return results
