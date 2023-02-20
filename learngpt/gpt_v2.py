from typing import Union
import torch
from .gpt_v1 import GPTVer1
from .head_v1 import HeadVer1


class GPTVer2(GPTVer1):

    def __init__(self, contextualizer: Union[HeadVer1, torch.nn.Module],
                 vocab_size: int, embed_size: int, block_size: int):
        super().__init__(vocab_size, block_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.contextualizer = contextualizer
        self.token_embedding_table = torch.nn.Embedding(vocab_size, embed_size)  # (|V|, C)
        self.lm_head = torch.nn.Linear(embed_size, vocab_size)  # (C, |V|)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        # --- TODO 2 --- #
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T) ->  (B, T, C)
        x = self.contextualizer(tok_emb)  # (B, T, C) ->  (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) @ (B, T, |V|) -> (B, T, |V|)
        # ------------- #
        return logits
