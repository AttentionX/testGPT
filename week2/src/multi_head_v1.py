from typing import Optional
import torch
from torch.nn import functional as F


class HeadVer4(torch.nn.Module):
    """
    single-head self-attention
    (redefining this here in case you haven't finished week1/src/head_v4.py)
    """
    def __init__(self, block_size: int, embed_size: int, head_size: int):
        super().__init__()
        self.key = torch.nn.Linear(embed_size, head_size, bias=False)  # (C, C)
        self.query = torch.nn.Linear(embed_size, head_size, bias=False)  # (C, C)
        self.value = torch.nn.Linear(embed_size, head_size, bias=False)  # (C, C)
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
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        self.var = wei.var().detach()  # log the variance of the attention scores right after scaling with 1/sqrt(d_k)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T), a.k.a "attentions"
        self.wei = wei.detach()  # log the final weights
        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadVer1(torch.nn.Module):
    """
    multi-head self attention (slow)
    """

    def __init__(self, block_size: int, embed_size: int, n_heads: int):
        """
        :param block_size: 32 (문장 속 토큰의 개수)
        :param embed_size: 512 ( 히든 벡처 차원의 크기)
        :param n_heads: 8 (multi-head attention의 헤드 개수)
        ...
        head_size = embed_size / n_heads
        """
        super().__init__()
        assert embed_size % n_heads == 0
        head_size = embed_size // n_heads
        # stack HeadVer4 as a ModuleList of Modules
        self.heads = torch.nn.ModuleList([
            HeadVer4(block_size, embed_size, head_size)
            for _ in range(n_heads)
        ])
        self.proj = torch.nn.Linear(head_size * n_heads, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # torch.cat vs. torch.concat; what should I use?
        out = torch.concat([head(x) for head in self.heads], dim=-1)  # (B, T, C) ->  (B, T, head_size * n_heads)
        out = self.proj(out)  # (B, T, head_size * n_heads) * (head_size * n_heads, C) ->  (B, T, C)
        return out
