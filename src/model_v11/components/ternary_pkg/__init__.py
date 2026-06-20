from .quantizers import *
from .linear import (
    TernaryLinear,
    TernaryLinearBTG,
    TernaryLinearBTG_PWST,
    TernaryLinearTATGR,
    MagnitudePrunedTernaryQuantize,
    pwst_forward_vectorized,
    pwst_backward_grad_input,
    invalidate_packed_cache_model,
)
from .variants import (
    DeferredTernaryLinear,
    TernaryLinearInteger,
    merge_all_deferred_deltas,
    convert_model_to_deferred,
    get_deferred_stats,
)
from .utils import (
    apply_antiwindup,
    compute_angular_boundary_reg,
    compute_tdo_loss,
    compute_weight_entropy,
    compute_commitment_rate,
    compute_margin_mask,
    init_gradient_recyclers,
    apply_gradient_recycling,
)
from .modifiers import (
    enable_dendritic_gating,
    enable_sigma_delta,
    enable_bias_correction,
    apply_bias_correction,
    enable_block_diagonal,
    get_block_diagonal_stats,
    enable_tpsq,
    enable_tcd,
    enable_mdts,
    enable_gumbel_noise,
    enable_stp,
    enable_ctce,
    enable_dual_rate,
    enable_gaste,
    enable_rlec,
)
from .monitors import TernaryFlipRateTracker
from .losses import (
    compute_sca_sparsity_loss,
    compute_sca_sparsity_loss_simple,
)
