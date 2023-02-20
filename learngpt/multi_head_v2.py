import torch


class MultiHeadVer2(torch.nn.Module):
    """
    multi-head self attention (fast)
    """

    def __init__(self, block_size: int, embed_size: int, n_heads: int):
        """
        :param block_size: 32 (문장 속 토큰의 개수)
        :param embed_size: 512 ( 히든 벡처 차원의 크기)
        :param n_heads: 8 (multi-head attention 의 헤드 개수)
        ...
        head_size = embed_size / n_heads
        """
        super().__init__()
        assert embed_size % n_heads == 0
        self.n_heads = n_heads
        self.head_size = embed_size // n_heads
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # --- layers to optimise --- #
        self.query = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.key = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.value = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.proj = torch.nn.Linear(embed_size, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return:
        """
        # --- TODO --- #
        B, T, C = x.shape
        q = self.query(x)  # (B, T, C) @ (C, C) -> (B, T, C)
        k = self.key(x)  # (B, T, C) @ (C, C) -> (B, T, C)
        v = self.value(x)  # (B, T, C) @ (C, C) -> (B, T, C)
        # split q, k, v into multi-heads
        q = q.contiguous().view(B, T, self.n_heads, self.head_size)  # (B, T, C) -> (B, T, n_heads, head_size)
        k = k.contiguous().view(B, T, self.n_heads, self.head_size)  # (B, T, C) -> (B, T, n_heads, head_size)
        v = v.contiguous().view(B, T, self.n_heads, self.head_size)  # (B, T, C) -> (B, T, n_heads, head_size)
        # make q, k, v matmul-compatible
        q = q.transpose(1, 2)  # (B, T, n_heads, head_size) -> (B, n_heads, T, head_size)
        k = k.transpose(1, 2)  # (B, T, n_heads, head_size) -> (B, n_heads, T, head_size)
        v = v.transpose(1, 2)  # (B, T, n_heads, head_size) -> (B, n_heads, T, head_size)
        # compute attention scores; (Q * K^T: compute query-key similarities)
        # (..., T, head_size) @ (..., head_size, T) ->  (..., T, T)
        sims = q @ k.transpose(-2, -1) * (C ** -0.5)
        # apply padding mask and/or subsequent mask
        sims = sims.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # softmax(Q * K^T / sqrt(d_k)): normalise the sims over keys
        attentions = torch.softmax(sims, dim=-1)
        # softmax(Q * K^T / sqrt(d_k)) * V: soft-align values with respect to each query
        # (..., T, T) @ (..., T, head_size) -> (..., T, head_size)
        alignments = attentions @ v
        # (B, n_heads, T, head_size) --transpose--> (B, T, n_heads, head_size) --concat--> (B, T, C)
        concats = alignments.transpose(1, 2).contiguous().view(B, T, C)
        # aggregate concatenations
        out = self.proj(concats)  # (B, T, C) @ (C, C) -> (B, T, C)
        # ------------ #
        return out
