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
        self.query = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.key = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.value = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.proj = torch.nn.Linear(embed_size, embed_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

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
        # make q, k and v matmul-compatible
        q = q.transpose(1, 2)  # (B, T, n_heads, head_size) -> (B, n_heads, T, head_size)
        k = k.transpose(1, 2)  # (B, T, n_heads, head_size) -> (B, n_heads, T, head_size)
        v = v.transpose(1, 2)  # (B, T, n_heads, head_size) -> (B, n_heads, T, head_size)
        # compute attention scores
        # Q * K^T:  compute query-key similarities
        # (B, n_heads, T, head_size), (B, n_heads, T, head_size) ->  (B, n_heads, T, T)
        wei = torch.einsum("...qh,...kh->...qk", q, k)
        # Q * K^T / sqrt(d_k): down-scale similarities to prevent gradient vanishing
        wei = wei * C ** -0.5
        # apply padding mask and/or subsequent mask
        wei = wei.masked_fill(self.tril[:T, :T], float("-inf"))
        # softmax(Q * K^T / sqrt(d_k)): normalise the sims over keys
        wei = torch.softmax(wei, dim=-1)
        # softmax(Q * K^T / sqrt(d_k)) * V: soft-align values with respect to each query
        # (B, n_heads, T, T),  (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)
        wei = torch.einsum("...qv,...vh->...qh", wei, v)
        wei = wei.transpose(1, 2)  # (B, n_heads, T, head_size) -> (B, T, n_heads, head_size)
        wei = wei.contiguous().view(B, T, C)  # (B, T, n_heads, head_size) -> (B, T, C)
        out = self.proj(wei)  # (B, T, C) @ (C, C) -> (B, T, C)
        # ------------ #
        return out
