import torch
from torch.nn import functional as F
from typing import Optional


class HeadVer4(torch.nn.Module):
    """
    single-head self-attention
    (redefining this here in case you haven't completed week1/src/head_v4.py)
    """
    def __init__(self, block_size: int, embed_size: int, head_size: int):
        super().__init__()
        self.key = torch.nn.Linear(embed_size, head_size, bias=False)  # (C, head_size)
        self.query = torch.nn.Linear(embed_size, head_size, bias=False)  # (C, head_size)
        self.value = torch.nn.Linear(embed_size, head_size, bias=False)  # (C, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.var: Optional[torch.Tensor] = None
        self.attentions: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        :param x: (B, T, C)
        :param test: for testing purposes
        :return: (B, T, C)
        """
        B, T, C = x.shape
        q = self.query(x)  # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        k = self.key(x)  # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        v = self.value(x)  # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        sims = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # log the variance of weights right after scaling with 1/sqrt(d_k)
        self.var = sims.var().detach()
        sims = sims.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        attentions = F.softmax(sims, dim=-1)  # normalize across features dim
        self.attentions = attentions.detach()  # log the attention scores
        # perform the weighted aggregation of the values (soft-align values)
        alignments = attentions @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return alignments
