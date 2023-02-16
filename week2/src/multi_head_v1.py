import torch
from .head_v4 import HeadVer4


class MultiHeadVer1(torch.nn.Module):
    """
    multi-head self attention (slow)
    """
    def __init__(self, block_size: int, embed_size: int, n_heads: int):
        """
        :param block_size: 32 (문장 속 토큰의 개수)
        :param embed_size: 512 ( 히든 벡터 차원의 크기)
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
        self.proj = torch.nn.Linear(embed_size, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # --- TODO 1-1 --- #
        raise NotImplementedError
        out = ...
        # ---------------- #
        return out
