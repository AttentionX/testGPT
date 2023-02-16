"""
adding LayerNorm to the block
"""
import torch
from typing import Union
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2
from .block_v2 import BlockVer2
from .layer_normalization import LayerNorm


class BlockVer3(BlockVer2):

    def __init__(self, head: Union[MultiHeadVer1, MultiHeadVer2], embed_size: int):
        super().__init__(head, embed_size)
        self.ln1 = LayerNorm(embed_size)
        self.ln2 = LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- TODO 3-2 --- #
        raise NotImplementedError
        x = ...
        # ---------------- #
        return x
