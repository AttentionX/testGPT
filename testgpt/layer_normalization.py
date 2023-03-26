"""
an implementation of layer normalization
"""
import torch


class LayerNorm(torch.nn.Module):
    """
    why do we need layer norm? - how should we test for this?
    - https://arxiv.org/pdf/1607.06450.pdf (Ba, Kiros & Hinton, 2016)
    but why use layerNorm instead of Batch norm?
    - https://paperswithcode.com/method/layer-normalization
    - " layer normalization does not impose any constraint on the size of the mini-batch and it can be used in the pure online regime with batch size 1."
    = i.e. independent of the batch (e.g. sequences of variable length, batch size, etc)
    but why scale & shift?
    - https://arxiv.org/pdf/1502.03167.pdf
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # --- learnable parameters --- #
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        # epsilon for numerical stability
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        objective: to normalize the input across feature dimension (standardize it to follow D(0, 1^2))
        how?:
        x = x - mean / std
        shifting & scaling: https://youtu.be/sxEqtjLC0aM?t=49
        why scale with gamma & beta?
        :param x: (B, T, C)
        :return: (B, T, C)
        """
        # --- TODO 3-1 --- #
        # compute the mean of x across feature dimension
        mean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)

        # center the distribution around zero by subtracting the distribution with mean
        """
        <subtracting a from each datapoint results in subtracting a from the mean>
        let Y = X - a
        then,
        E[Y] = E[X - a]
             = (Σ(xi - a)) / n
             = (Σxi - na) / n
             = (Σxi) / n - a
             = E[X] - a
        thus, E[Y] = E[X] - a
        ...
        if a = E[X]?
        then E[Y] = μ - μ = 0
        """
        new = x - mean  # (B, T, C)
        # compute the std of x across feature dimension
        std = x.std(dim=-1, keepdim=True)  # (B, T, 1)
        # shrink the distribution to have std of 1 by down-scaling it with std
        """
        <scaling each datapoint by b results in scaling the std by b>
        let Y = bX
        then,
        Var[Y] = Var[bX]
               = b^2 Var[X]
               = b^2 * Std[X]^2
        Std[Y] = sqrt(b^2 * Std[X]^2)
               = b * Std[X]
        thus,
        Std[Y] = b * Std[X] 
        ... 
        if b = 1 / Std[X] (down-scale by std)?
        then Std[Y] = 1 / Std[X] * Std[X] = 1
        """
        new = new / (std + self.eps)  # (B, T, C)

        # scale & shift with alpha & gamma
        """
         ... Note that simply normalizing each input of a layer may change what the
        layer can represent. For instance, normalizing the inputs of a sigmoid would constrain 
        them to the linear regime of the nonlinearity.(Ioffe & Szegedy, 2015)
        """
        new = self.gamma * new + self.beta  # (B, T, C)
        # ---------------- #
        return new



