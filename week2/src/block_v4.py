"""
adding dropout to the block
"""
import torch
from typing import Union
from .block_v3 import BlockVer3
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2


class BlockVer4(BlockVer3):

    def __init__(self, head: Union[MultiHeadVer1, MultiHeadVer2], embed_size: int, dropout: float):
        super().__init__(head, embed_size)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- TODO --- #
        # implement the residual connection
        
        # x = x + self.head(self.ln_1(x))  -> block에서는 dropout 포함 안됨!!!
        # x = x + self.ffwd(self.ln_2(x))  -> block에서는 dropout 포함 안됨!!!
        
        x = x + self.dropout_1(self.head(self.ln_1(x)))
        x = x + self.dropout_2(self.ffwd(self.ln_2(x)))
        # ------------ #
        return x
