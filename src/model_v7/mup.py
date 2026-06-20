import math
import re
import torch
import torch.nn as nn


def get_mup_param_groups(model, base_lr, base_width, current_width):
    """Compute muP-scaled parameter groups with width-aware learning rates.

    SSM layers get quadratic LR scaling (base_lr / m_d^2) because their
    recurrent state accumulates updates across time steps, amplifying the
    effective learning rate.  SWA (attention) layers get standard linear
    scaling (base_lr / m_d).
    """
    m_d = current_width / base_width

    # Determine which layer indices are SWA vs SSM
    swa_indices = set()
    if hasattr(model, 'config') and hasattr(model.config, 'swa_layer_indices'):
        swa_indices = set(model.config.swa_layer_indices)

    param_groups = {
        'a_matrix': [],
        'embedding': [],
        'router': [],
        'norm': [],
        'bias': [],
        'conv1d': [],
        'ssm_d': [],
        'ssm_hidden': [],   # SSM projection weights (quadratic LR)
        'swa_hidden': [],   # SWA projection weights (linear LR)
        'hidden': [],       # Fallback for non-layer params
    }

    # Regex to extract layer index from parameter name
    layer_pattern = re.compile(r'^layers\.(\d+)\.')

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'A_log' in name:
            param_groups['a_matrix'].append(param)
        elif 'embedding' in name:
            param_groups['embedding'].append(param)
        elif 'router' in name:
            param_groups['router'].append(param)
        elif '.norm.' in name or 'group_norm' in name or name == 'final_norm.weight':
            param_groups['norm'].append(param)
        elif name.endswith('.bias'):
            param_groups['bias'].append(param)
        elif 'conv1d' in name:
            param_groups['conv1d'].append(param)
        elif name.endswith('.D'):
            param_groups['ssm_d'].append(param)
        else:
            # Determine if this param belongs to an SSM or SWA layer
            match = layer_pattern.match(name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx in swa_indices:
                    param_groups['swa_hidden'].append(param)
                else:
                    param_groups['ssm_hidden'].append(param)
            else:
                param_groups['hidden'].append(param)

    groups_list = []

    if param_groups['a_matrix']:
        groups_list.append({
            'params': param_groups['a_matrix'],
            'lr': base_lr * 0.1,
            'weight_decay': 0.0,
            'name': 'a_matrix',
        })

    if param_groups['embedding']:
        groups_list.append({
            'params': param_groups['embedding'],
            'lr': base_lr / math.sqrt(m_d),
            'weight_decay': 0.0,
            'name': 'embedding',
        })

    if param_groups['router']:
        groups_list.append({
            'params': param_groups['router'],
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'router',
        })

    if param_groups['norm']:
        groups_list.append({
            'params': param_groups['norm'],
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'norm',
        })

    if param_groups['bias']:
        groups_list.append({
            'params': param_groups['bias'],
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'bias',
        })

    if param_groups['conv1d']:
        groups_list.append({
            'params': param_groups['conv1d'],
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'conv1d',
        })

    if param_groups['ssm_d']:
        groups_list.append({
            'params': param_groups['ssm_d'],
            'lr': base_lr / m_d,
            'weight_decay': 0.0,
            'name': 'ssm_d',
        })

    # SSM hidden: quadratic scaling -- recurrent state amplifies updates
    if param_groups['ssm_hidden']:
        groups_list.append({
            'params': param_groups['ssm_hidden'],
            'lr': base_lr / (m_d ** 2),
            'weight_decay': 0.1,
            'name': 'ssm_hidden',
        })

    # SWA hidden: standard linear scaling
    if param_groups['swa_hidden']:
        groups_list.append({
            'params': param_groups['swa_hidden'],
            'lr': base_lr / m_d,
            'weight_decay': 0.1,
            'name': 'swa_hidden',
        })

    # Non-layer hidden (lm_head, etc.): standard linear scaling
    if param_groups['hidden']:
        groups_list.append({
            'params': param_groups['hidden'],
            'lr': base_lr / m_d,
            'weight_decay': 0.1,
            'name': 'hidden',
        })

    print("\n[muP Parameter Groups Summary]")
    print(f"  m_d = {current_width} / {base_width} = {m_d:.4f}")
    total_params = 0
    for group in groups_list:
        count = sum(p.numel() for p in group['params'])
        total_params += count
        print(f"  {group['name']:<15} | params: {count:>10,} | lr: {group['lr']:.2e} | wd: {group.get('weight_decay', 0.1):.3f}")
    print(f"  {'TOTAL':<15} | params: {total_params:>10,}")

    return groups_list


def apply_mup_init(model, base_width):
    """Re-initialize model parameters according to muP rules.

    Args:
        model: ModelV2 instance
        base_width: Reference width (used for logging only)
    """
    reinitialized = 0
    skipped = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        skip_categories = ['A_log', 'D', 'dt_proj.bias', 'conv1d', 'norm', 'router']
        skip = any(cat in name for cat in skip_categories) or param.dim() == 1

        if skip:
            skipped += 1
            continue

        if 'embedding.weight' in name:
            nn.init.normal_(param, std=0.02)
            reinitialized += 1
        elif param.dim() >= 2:
            fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
            nn.init.normal_(param, std=1.0 / math.sqrt(fan_in))
            reinitialized += 1
        else:
            skipped += 1

    print(f"\n[muP Initialization Summary]")
    print(f"  Reinitialized: {reinitialized} parameters")
    print(f"  Skipped: {skipped} parameters (A_log, D, norms, routers, 1D params)")


def apply_output_scaling(model, base_width, current_width):
    """Apply output scaling for muP transfer: logits *= 1/m_d.

    Args:
        model: ModelV2 instance
        base_width: Reference width
        current_width: Current model width
    """
    m_d = current_width / base_width
    model.output_multiplier = 1.0 / m_d
    print(f"\n[Output Scaling]")
    print(f"  output_multiplier = 1/{m_d:.4f} = {model.output_multiplier:.4f}")
