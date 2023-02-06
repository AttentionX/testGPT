from typing import Union
from . import GPTVer1, HeadVer1, HeadVer2, HeadVer3, HeadVer4
import torch
import torch.nn as nn


class GPTVer2(GPTVer1):

    def __init__(self, head: Union[HeadVer1, HeadVer2, HeadVer3, HeadVer4],
                 vocab_size: int, embed_size: int, block_size: int):
        super().__init__(vocab_size, block_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.head = head
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)  # (|V|, C)
        self.lm_head = nn.Linear(embed_size, vocab_size)  # (C, |V|)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        # --- TODO 2 --- #
        logits = ...
        raise NotImplementedError
        # -------------- #
        return logits