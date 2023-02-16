"""
an implementation of layer normalization
"""
import torch


class LayerNorm(torch.nn.Module):
    """
    why do we need layer norm? - how should we test for this?
    - https://arxiv.org/pdf/1607.06450.pdf (Ba, Kiros & Hinton, 2016)
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # --- learnable parameters --- #
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # --- TODO 3-1 --- #
        raise NotImplementedError
        # ---------------- #
