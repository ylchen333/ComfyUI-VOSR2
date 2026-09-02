# Vendored from VOSR's models/rmsnorm.py (upstream: Meta Llama 2 RMSNorm),
# trimmed to the single class LightningDiT actually uses. The upstream file
# also carries an unrelated fairscale-based Transformer that VOSR2 never
# constructs and that would pull in a model-parallel dependency for nothing.
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
