"""
adding dropout to the block
"""
import torch
from typing import Union
from .block_v3 import BlockVer3
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2
from .feed_forward import FeedForward


class BlockVer4(BlockVer3):

    def __init__(self, head: Union[MultiHeadVer1, MultiHeadVer2], embed_size: int, dropout: float):
        super().__init__(head, embed_size)
        # --- TODO 4 --- #
        self.head = ...
        self.ffwd = ...
        raise NotImplementedError
        # -------------- #
