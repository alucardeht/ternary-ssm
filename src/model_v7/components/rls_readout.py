import torch
import torch.nn as nn


class RLSReadout:
    """
    RLS (Recursive Least Squares) readout para o lm_head.
    Trata o SSM + rede inteira como reservatório fixo.
    Atualiza W_out sem backpropagation.

    Referência: FORCE learning (Sussillo & Abbott 2009), RLS para reservatórios ternários (Liu et al. 2022).
    Update: k = P @ h / (λ + h @ P @ h)
            W_out -= outer(error, k)   # shape: (vocab_size, d_model)
            P = (P - outer(k, P @ h)) / λ
    """

    def __init__(self, d_in: int, d_out: int, forgetting_factor: float = 0.99, delta_init: float = 0.01):
        self.d_in = d_in
        self.d_out = d_out
        self.forgetting_factor = forgetting_factor
        self.P = torch.eye(d_in) / delta_init
        self._n_updates = 0

    def to(self, device):
        self.P = self.P.to(device)
        return self

    def update(self, h: torch.Tensor, y_target: torch.Tensor, W_out: torch.nn.Parameter):
        """Single-token RLS update of W_out readout matrix."""
        with torch.no_grad():
            lam = self.forgetting_factor

            # FIX 4: normalise h to prevent correlated h_t from starving denominator
            h_n = h / (h.norm() + 1e-8)

            Ph = self.P @ h_n
            denom = lam + h_n @ Ph
            if abs(denom.item()) < 1e-12:
                return
            k = Ph / denom

            y_pred = W_out.data @ h_n
            error = y_pred - y_target
            W_out.data -= torch.outer(error, k)
            self.P = (self.P - torch.outer(k, Ph)) / lam

            # FIX 2: trace clipping — prevent P from growing unboundedly
            P_trace = torch.diagonal(self.P).sum()
            P_max = self.d_in * 10.0
            if P_trace > P_max:
                self.P.mul_(P_max / P_trace)

            self._n_updates += 1
            if self._n_updates % 100 == 0:
                print(f"[RLS] update={self._n_updates} trace(P)={P_trace:.1f} |k|={k.norm():.4f} |err|={error.norm():.4f}", flush=True)

    def update_batch(self, h_seq: torch.Tensor, y_seq: torch.Tensor, W_out: torch.nn.Parameter):
        """
        h_seq: (seq_len, d_in)
        y_seq: (seq_len, d_out) — targets one-hot ou distribuição
        Aplica update sequencialmente (correto para RLS causal).
        """
        for t in range(h_seq.shape[0]):
            self.update(h_seq[t], y_seq[t], W_out)

    def reset(self, delta_init: float = 0.01):
        """Reinicializa P (útil no início de cada epoch)."""
        device = self.P.device
        self.P = torch.eye(self.d_in, device=device) / delta_init
        self._n_updates = 0

    @property
    def n_updates(self):
        return self._n_updates
