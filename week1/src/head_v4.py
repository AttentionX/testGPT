from typing import Optional
import torch
from torch.nn import functional as F


class HeadVer4(torch.nn.Module):
    """ i.e. one head of self-attention """
    def __init__(self, block_size: int, embed_size: int):
        super().__init__()
        self.key = torch.nn.Linear(embed_size, embed_size, bias=False)  # (C, C)
        self.query = torch.nn.Linear(embed_size, embed_size, bias=False)  # (C, C)
        self.value = torch.nn.Linear(embed_size, embed_size, bias=False)  # (C, C)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.var: Optional[torch.Tensor] = None
        self.wei: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, test: bool = False) -> torch.Tensor:
        """
        Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        :param x: (B, T, C)
        :param test: for testing purposes
        :return: (B, T, C)
        """
        B, T, C = x.shape
        if test:
            q = torch.randn(B, T, C)  # (B, T, C)
            k = torch.randn(B, T, C)  # (B, T, C)
            v = torch.randn(B, T, C)  # (B, T, C)
        else:
            q = self.query(x)  # (B, T, C)
            k = self.key(x)  # (B, T, C)
            v = self.value(x)  # (B, T, C)
        # --- TODO 5 --- #
        wei = ...
        self.var = wei.var().detach()  # log the variance of the attention scores right after scaling with 1/sqrt(d_k)
        wei = ...
        self.wei = wei.detach()  # log the final weights
        out = ...
        raise NotImplementedError
        # ------------ #
        return out
