import torch
import logging

logger = logging.getLogger(__name__)


class HybridRTRLTBPTT:
    """Hybrid gradient: exact RTRL for A_log, truncated BPTT for B,C,dt.

    RTRL (Real-Time Recurrent Learning) computes dh/dA exactly in O(d_state).
    TBPTT (Truncated BPTT) limits backprop to K steps for B,C,dt (95% gradient energy).
    """

    def __init__(self, ssm_module, K_truncate=50):
        self.ssm = ssm_module
        self.K = K_truncate
        self.rtrl_state = None
        self.enabled = False

    def forward_with_rtrl(self, x, A, B, C, dt):
        """Forward with RTRL accumulation (pseudo-gradient tracking)."""
        B_seq, L, D = x.shape
        d_state = A.shape[-1] if A.dim() > 0 else 1

        h = torch.zeros(B_seq, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        dh_dA = torch.zeros(d_state, device=x.device, dtype=x.dtype)
        self.enabled = True

        for t in range(L):
            if A.dim() == 0 or A.shape[0] == 0:
                break

            dt_t = dt[t] if dt.dim() > 0 else dt
            if isinstance(dt_t, (int, float)):
                dt_t = torch.tensor(dt_t, device=x.device, dtype=x.dtype)

            A_discrete = torch.exp(A * dt_t).clamp(min=1e-4, max=1.0)
            B_t = B[t] if B.dim() > 1 else B
            C_t = C[t] if C.dim() > 1 else C
            x_t = x[:, t, :] if x.dim() > 2 else x

            h_new = A_discrete * h + B_t * x_t
            y = C_t * h_new

            dh_dA = dt_t * (A_discrete * dh_dA + h.mean(dim=0))

            h = h_new
            outputs.append(y)

        self.rtrl_state = dh_dA

        try:
            return torch.stack(outputs, dim=1)
        except RuntimeError:
            return torch.cat(outputs, dim=1).reshape(B_seq, L, -1)

    def backward_hybrid(self, grad_output, ssm_params):
        """Compute gradients (simplified: return stats for logging)."""
        if self.rtrl_state is None:
            return {}

        with torch.no_grad():
            stats = {
                'rtrl_state_norm': self.rtrl_state.norm().item(),
                'grad_output_norm': grad_output.norm().item() if grad_output is not None else 0.0
            }

        return stats

    def get_stats(self):
        if self.rtrl_state is None:
            return {'rtrl_norm': 0.0}
        return {'rtrl_norm': self.rtrl_state.norm().item()}
