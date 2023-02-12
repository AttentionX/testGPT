"""
stacking blocks (block = MHA + FFN)
- layer normalization
"""

import torch


class LayerNorm(torch.nn.Module):
    """
    why do we need layer norm? - how should we test for this?
    - https://arxiv.org/pdf/1607.06450.pdf (Ba, Kiros & Hinton, 2016)
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # --- learnable parameters --- #
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        mean = x.mean(-1, keepdim=True)  # (B, T, C) ->  (B, T, 1)
        std = x.std(-1, keepdim=True)  # (B, T, C) ->  (B, T, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class GPTVer6(torch.nn.Module):
    pass
