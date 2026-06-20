import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class AsymmetricChunkWrapper(nn.Module):
    """Wrap model with different chunk sizes for SSM vs SWA layers.

    SSM: Large chunks (d_state*d_model boundary overhead reduced).
    SWA: Small chunks (window^2*d_model memory limited).
    """

    def __init__(self, model, chunk_ssm=512, chunk_attn=128, enable_logging=False):
        super().__init__()
        self.model = model
        self.chunk_ssm = chunk_ssm
        self.chunk_attn = chunk_attn
        self.enable_logging = enable_logging

    def forward(self, x):
        B, L, D = x.shape

        outputs = []

        for i in range(0, L, self.chunk_ssm):
            chunk_end = min(i + self.chunk_ssm, L)
            x_chunk = x[:, i:chunk_end, :]

            h = x_chunk

            if hasattr(self.model, 'blocks'):
                for block in self.model.blocks:
                    if hasattr(block, 'ssm') and block.ssm is not None:
                        h = block.ssm(h)

                    if hasattr(block, 'swa') and block.swa is not None:
                        h = self._attn_chunked(block.swa, h, self.chunk_attn)

            outputs.append(h)

        result = torch.cat(outputs, dim=1)
        return result

    def _attn_chunked(self, attn_module, x, chunk_size):
        B, L, D = x.shape

        if L <= chunk_size:
            return attn_module(x)

        chunks = []
        for i in range(0, L, chunk_size):
            end = min(i + chunk_size, L)
            chunk_out = attn_module(x[:, i:end, :])
            chunks.append(chunk_out)

        return torch.cat(chunks, dim=1)
