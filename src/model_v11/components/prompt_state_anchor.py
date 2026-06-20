import torch
import torch.nn as nn


class PromptStateAnchor(nn.Module):
    def __init__(self, d_model, gamma_init=0.1, decay_mode='fixed'):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.decay_mode = decay_mode
        if decay_mode == 'exponential':
            self.log_rho = nn.Parameter(torch.zeros(1))
        self._logged = False

    def forward(self, x, prompt_state, prompt_len):
        prompt_state = prompt_state.detach()
        B, T, d = x.shape
        device = x.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        response_mask = positions >= prompt_len.unsqueeze(1)

        injection = prompt_state.unsqueeze(1) * self.gamma

        if self.decay_mode == 'exponential':
            rho = torch.sigmoid(self.log_rho)
            offset = (positions - prompt_len.unsqueeze(1)).clamp(min=0).float()
            decay = rho ** offset
            injection = injection * decay.unsqueeze(-1)

        injection = injection * response_mask.unsqueeze(-1).float()

        if not self._logged:
            print(f"[ANCHOR] active | gamma={self.gamma.item():.4f} mode={self.decay_mode}", flush=True)
            self._logged = True

        return x + injection
