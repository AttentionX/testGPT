import torch
from typing import Union
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2
from .block_v1 import BlockVer1


class BlockVer2(BlockVer1):
    """ Transformer block: communication followed by computation """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # --- TODO --- #
        # implement the residual connection
        x = x + self.head(x)
        x = x + self.ffwd(x)
        # ------------ #
        return x
