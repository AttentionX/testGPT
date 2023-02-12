"""
adding LayerNorm to the block
"""
import torch
from typing import Union
from .block_v1 import BlockVer1
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2


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
        # --- TODO --- #
        mean = x.mean(-1, keepdim=True)  # (B, T, C) ->  (B, T, 1)
        std = x.std(-1, keepdim=True)  # (B, T, C) ->  (B, T, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
        # ------------ #


class BlockVer3(BlockVer1):
    """ Transformer block: communication followed by computation """

    def __init__(self, head: Union[MultiHeadVer1, MultiHeadVer2], embed_size: int):
        super().__init__(head, embed_size)
        self.ln_1 = LayerNorm(embed_size)
        self.ln_2 = LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- TODO --- #
        # implement the residual connection
        x = x + self.head(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        # ------------ #
        return x
