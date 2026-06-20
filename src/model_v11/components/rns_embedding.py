"""RNS (Residue Number System) Embedding — 3000× fewer parameters than standard."""
import math
import torch
import torch.nn as nn


class RNSEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, moduli=(8, 9, 11, 13, 7)):
        super().__init__()
        M = math.prod(moduli)
        assert M >= vocab_size, f"Product of moduli {M} < vocab_size {vocab_size}"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.register_buffer('moduli', torch.tensor(moduli, dtype=torch.long))

        k = len(moduli)
        base_dim = d_model // k
        remainder = d_model % k

        self.tables = nn.ModuleList()
        self.dims = []
        for i, m in enumerate(moduli):
            dim = base_dim + (1 if i < remainder else 0)
            self.dims.append(dim)
            table = nn.Embedding(m, dim)
            nn.init.normal_(table.weight, std=0.02)
            self.tables.append(table)

        total_params = sum(m * d for m, d in zip(moduli, self.dims))
        print(f"[RNS-EMBED] moduli={moduli} M={M} params={total_params} (vs {vocab_size * d_model} standard)")

    @property
    def num_embeddings(self):
        return self.vocab_size

    def forward(self, token_ids):
        parts = []
        for i, (table, m) in enumerate(zip(self.tables, self.moduli)):
            r = token_ids % m.item()
            parts.append(table(r))
        return torch.cat(parts, dim=-1)
