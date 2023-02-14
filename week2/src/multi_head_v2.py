import torch


class MultiHeadVer2(torch.nn.Module):
    """
    multi-head self attention (fast)
    """

    def __init__(self, embed_size: int, n_heads: int):
        """
        :param block_size: 32 (문장 속 토큰의 개수)
        :param embed_size: 512 ( 히든 벡처 차원의 크기)
        :param n_heads: 8 (multi-head attention 의 헤드 개수)
        ...
        head_size = embed_size / n_heads
        """
        super().__init__()
        assert embed_size % n_heads == 0
        self.head_size = embed_size // n_heads
        self.query = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.key = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.value = torch.nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return:
        """
        # --- TODO --- #
        B, T, C = x.shape

        q = self.query(x)  # (N, L, H) * (H, H) -> (N, L, H)
        k = self.key(x)  # (N, L, H) * (H, H) -> (N, L, H)
        v = self.linear(x)  # (N, L, H) * (H, H) -> (N, L, H)
        # split q, k, v into multi-heads
        q = q.view(B, self.max_length, self.heads, self.head_size)  # (N, L, H) -> (N, L, heads, head_size)
        k = k.view(B, self.max_length, self.heads, self.head_size)  # (N, L, H) -> (N, L, heads, head_size)
        v = v.view(B, self.max_length, self.heads, self.head_size)  # (N, L, H) -> (N, L, heads, head_size)
        # make q, k and v matmul-compatible
        q = q.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        k = k.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        v = v.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)

        out  = ...
        # ------------ #
        return out
