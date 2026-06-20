from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_fisher_masks(
    model: nn.Module,
    fisher_path: Optional[str] = None,
    threshold: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute binary masks from Fisher diagonal.

    Positions where Fisher < threshold are 'sleeping zeros' — safe to modify.
    Returns dict mapping parameter name -> binary mask (1=safe, 0=protected).

    Args:
        model: The model (used for reference to extract parameter names)
        fisher_path: Path to Fisher diagonal checkpoint
        threshold: Fisher threshold below which positions are unprotected (safe)

    Returns:
        Dict mapping parameter name -> binary mask (1=safe to modify, 0=protected)
    """
    masks = {}

    if fisher_path is None:
        return masks

    fisher_dict = torch.load(fisher_path, map_location='cpu')

    for name, fisher_param in fisher_dict.items():
        if isinstance(fisher_param, torch.Tensor):
            mask = (fisher_param < threshold).float()
            masks[name] = mask

    return masks


def apply_fisher_masks(
    adapter: 'ATMAdapter',
    fisher_masks: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    """Register gradient hooks on ATM parameters to zero gradients at protected positions.

    Args:
        adapter: The ATMAdapter instance containing ATM modules
        fisher_masks: Dict mapping parameter names to binary masks

    Returns:
        Dict with masking statistics (protected_params, total_params, protected_pct)
    """
    if not fisher_masks:
        return {'protected_params': 0, 'total_params': 0, 'protected_pct': 0.0}

    total_params = 0
    total_protected = 0
    masked_layers = 0

    for layer_name, atm_module in adapter._atm_modules.items():
        matching_masks = {k: v for k, v in fisher_masks.items() if k in layer_name}

        if not matching_masks:
            continue

        mask = next(iter(matching_masks.values()))

        if mask.shape != atm_module.atm_weight.shape:
            continue

        masked_layers += 1
        device_mask = mask.to(atm_module.atm_weight.device)

        def hook_fn(grad, m=device_mask):
            return grad * m

        atm_module.atm_weight.register_hook(hook_fn)

        total_params += atm_module.atm_weight.numel()
        total_protected += (~(device_mask.bool())).sum().item()

    protected_pct = 100.0 * total_protected / max(total_params, 1)

    return {
        'masked_layers': masked_layers,
        'protected_params': total_protected,
        'total_params': total_params,
        'protected_pct': protected_pct,
    }


def compute_projection_bases(
    model: nn.Module,
    calibration_loader: DataLoader,
    n_batches: int = 25,
    variance_threshold: float = 0.9,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """Compute SVD-based projection bases from base model activations.

    For each layer, collects activations on calibration data, runs SVD,
    and keeps top-k singular vectors that explain >= variance_threshold of variance.

    Args:
        model: The model to collect activations from
        calibration_loader: DataLoader for calibration data
        n_batches: Number of batches to process
        variance_threshold: Cumulative variance threshold for keeping singular vectors
        device: Device to use

    Returns:
        Dict mapping layer_name -> U_k (principal directions to protect)
    """
    model.eval()
    activations_dict: Dict[str, list] = {}
    handles = []

    def make_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                acts = output.detach().cpu()
                if len(acts.shape) > 2:
                    acts = acts.view(-1, acts.shape[-1])
                if layer_name not in activations_dict:
                    activations_dict[layer_name] = []
                activations_dict[layer_name].append(acts)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    with torch.no_grad():
        batch_count = 0
        for batch_input, _ in calibration_loader:
            batch_input = batch_input.to(device)
            model(batch_input)
            batch_count += 1
            if batch_count >= n_batches:
                break

    for handle in handles:
        handle.remove()

    projection_bases = {}

    for layer_name, acts_list in activations_dict.items():
        if not acts_list:
            continue

        acts = torch.cat(acts_list, dim=0)

        if acts.shape[0] < 10:
            continue

        acts_centered = acts - acts.mean(dim=0, keepdim=True)

        U, S, _ = torch.linalg.svd(acts_centered, full_matrices=False)

        cumsum_var = torch.cumsum(S ** 2, dim=0)
        cumsum_var = cumsum_var / cumsum_var[-1]
        k = (cumsum_var < variance_threshold).sum().item() + 1
        k = min(k, U.shape[1])

        U_k = U[:, :k]
        projection_bases[layer_name] = U_k.to(device)

    return projection_bases


def apply_gradient_projection(
    adapter: 'ATMAdapter',
    projection_bases: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    """Register gradient hooks to project ATM gradients into null space of base activations.

    Args:
        adapter: The ATMAdapter instance containing ATM modules
        projection_bases: Dict mapping layer_name -> projection basis U_k

    Returns:
        Dict with projection statistics (projected_layers, avg_basis_dim)
    """
    if not projection_bases:
        return {'projected_layers': 0, 'avg_basis_dim': 0.0}

    projected_layers = 0
    total_dims = 0

    for layer_name, atm_module in adapter._atm_modules.items():
        matching_bases = {k: v for k, v in projection_bases.items() if k in layer_name}

        if not matching_bases:
            continue

        U_k = next(iter(matching_bases.values()))
        U_k = U_k.to(atm_module.atm_weight.device)

        projected_layers += 1
        total_dims += U_k.shape[1]

        original_shape = atm_module.atm_weight.shape

        def make_projection_hook(basis, shape):
            def hook_fn(grad):
                if grad is None:
                    return None

                grad_flat = grad.view(-1, shape[-1])
                grad_proj = grad_flat @ (torch.eye(basis.shape[0], device=grad.device) - basis @ basis.T)
                grad_reprojected = grad_proj.view(shape)

                return grad_reprojected

            return hook_fn

        atm_module.atm_weight.register_hook(make_projection_hook(U_k, original_shape))

    avg_basis_dim = total_dims / max(projected_layers, 1)

    return {
        'projected_layers': projected_layers,
        'avg_basis_dim': avg_basis_dim,
    }


def setup_anti_forgetting(
    model: nn.Module,
    adapter: 'ATMAdapter',
    fisher_path: Optional[str] = None,
    calibration_loader: Optional[DataLoader] = None,
    fisher_threshold: float = 1e-6,
    variance_threshold: float = 0.9,
    use_fisher: bool = True,
    use_projection: bool = True,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Set up all anti-forgetting mechanisms.

    Args:
        model: The model to protect
        adapter: The ATMAdapter instance
        fisher_path: Path to Fisher diagonal checkpoint (for Fisher masking)
        calibration_loader: DataLoader for calibration (for PEGP projection)
        fisher_threshold: Fisher threshold for sleeping zero identification
        variance_threshold: Cumulative variance threshold for SVD truncation
        use_fisher: Whether to apply Fisher masking
        use_projection: Whether to apply gradient projection
        device: Device to use

    Returns:
        Dict with statistics from both mechanisms
    """
    stats = {
        'fisher': {},
        'projection': {},
    }

    if use_fisher and fisher_path is not None:
        fisher_masks = compute_fisher_masks(model, fisher_path, fisher_threshold)
        fisher_stats = apply_fisher_masks(adapter, fisher_masks)
        stats['fisher'] = fisher_stats

        if fisher_stats.get('protected_params', 0) > 0:
            print(
                f"Fisher masking applied: {fisher_stats['masked_layers']} layers | "
                f"{fisher_stats['protected_params']:,}/{fisher_stats['total_params']:,} params protected "
                f"({fisher_stats['protected_pct']:.1f}%)"
            )

    if use_projection and calibration_loader is not None:
        projection_bases = compute_projection_bases(
            model, calibration_loader, variance_threshold=variance_threshold, device=device
        )
        projection_stats = apply_gradient_projection(adapter, projection_bases)
        stats['projection'] = projection_stats

        if projection_stats.get('projected_layers', 0) > 0:
            print(
                f"Gradient projection applied: {projection_stats['projected_layers']} layers | "
                f"avg basis dim {projection_stats['avg_basis_dim']:.1f}"
            )

    return stats
