"""
adding LayerNorm to the block
"""
import torch
from typing import Union
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2
from .block_v2 import BlockVer2
from .block_v3_ln import LayerNorm


class BlockVer3(BlockVer2):

    def __init__(self, head: Union[MultiHeadVer1, MultiHeadVer2], embed_size: int):
        super().__init__(head, embed_size)
        self.ln_1 = LayerNorm(embed_size)
        self.ln_2 = LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- TODO --- #
        x = x + self.head(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        # ------------ #
        return x
