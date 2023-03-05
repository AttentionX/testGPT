import torch


class FeedForward(torch.nn.Module):
    """
    (Vaswani et. al, pg. 5)
    FFN(x) = max(0, xW1 + b1 )W2 + b2
    """
    def __init__(self, embed_size: int):
        super().__init__()
        # --- TODO 2-1 --- #
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_size, 4 * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_size, embed_size)
        )
        # ---------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        return self.net(x)
