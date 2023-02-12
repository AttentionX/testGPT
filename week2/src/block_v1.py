import torch
from typing import Union
from .multi_head_v1 import MultiHeadVer1
from .multi_head_v2 import MultiHeadVer2


class FeedForward(torch.nn.Module):

    def __init__(self, embed_size: int):
        super().__init__()
        # --- TODO --- #
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_size, 4 * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_size, embed_size),
        )
        # ------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BlockVer1(torch.nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, head: Union[MultiHeadVer1, MultiHeadVer2], embed_size: int):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.head = head
        self.ffwd = FeedForward(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # --- TODO --- #
        # implement the residual connection
        x = self.head(x)
        x = self.ffwd(x)
        # ------------ #
        return x
