import torch.nn


class FeedForward(torch.nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        # --- TODO 2-1 --- #
        self.net = ...
        raise NotImplementedError
        # ---------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        return self.net(x)
