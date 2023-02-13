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

        out  = ...
        # ------------ #
        return out
