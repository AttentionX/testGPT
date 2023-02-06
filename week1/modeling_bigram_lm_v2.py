from typing import Union
from modeling_head_v1 import HeadVer1
from modeling_head_v2 import HeadVer2
from modeling_head_v3 import HeadVer3
from modeling_head_v4 import HeadVer4
import torch
import torch.nn as nn
from modeling_bigram_lm_v1 import BigramLMVer1


class BigramLMVer2(BigramLMVer1):

    def __init__(self, head: Union[HeadVer1, HeadVer2, HeadVer3, HeadVer4], vocab_size: int, embed_size: int):
        super().__init__(vocab_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.head = head
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)  # (|V|, C)
        self.lm_head = nn.Linear(embed_size, vocab_size)  # (C, |V|)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :param pos: if True, use positional encoding
        :return: logits (B, T, |V|)
        """
        B, T = idx.shape
        C = self.token_embedding_table.weight.shape[1]
        # --- TODO --- #
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T) ->  (B, T, C)
        pos_emb = self.pos_encodings(T, C).to(idx.device)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.head(x)  # apply one head  of self-attention. (B, T, C)
        logits = self.lm_head(x)  # (B, T, |V|)
        # ------------ #
        return logits

    @staticmethod
    def pos_encodings(block_size: int, embed_size: int) -> torch.Tensor:
        """
        :param block_size: length of the sequence (T)
        :param embed_size: number of embedding dimensions (C)
        :return: (L, H)
        """
        # --- TODO --- #
        positions = torch.arange(block_size).view(-1, 1)  # -> (L)
        freqs = 0.0001 ** (torch.arange(embed_size)[::2] / embed_size).view(1, -1)  # (,) -> (H)
        encodings = torch.zeros(size=(block_size, embed_size))  # (L, H)
        encodings[:, ::2] = torch.sin(freqs * positions)  # evens = sin
        encodings[:, 1::2] = torch.cos(freqs * positions)
        # ------------ #
        return encodings
