import copy
import torch
from torch.nn import functional as F


class HeadVer1(torch.nn.Module):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x (B, T, C)
        :return: out (B, T, C)
        """
        B, T, C = x.shape
        out = torch.zeros((B, T, C)).to(x.device)
        for b in range(B):
            for t in range(T):
                xprev = x[b, :t + 1]
                out[b, t] = torch.mean(xprev, 0)
        return out


class HeadVer2(torch.nn.Module):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return:
        """
        T = x.shape[1]
        wei = torch.tril(torch.ones(T, T)).to(x.device)
        wei = wei / wei.sum(1, keepdim=True)
        # perform the weighted aggregation of the values
        out = wei @ x  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class HeadVer3(torch.nn.Module):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        tril = torch.tril(torch.ones(T, T))
        wei = torch.zeros((T, T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ x
        return out


class HeadVer4(torch.nn.Module):
    """ i.e. one head of self-attention """
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, n_embd, bias=False)  # (C, C)
        self.query = torch.nn.Linear(n_embd, n_embd, bias=False)  # (C, C)
        self.value = torch.nn.Linear(n_embd, n_embd, bias=False)  # (C, C)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.var = None
        self.wei = None

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        if debug:
            k = torch.randn(B, T, C)  # (B, T, C)
            q = torch.randn(B, T, C)  # (B, T, C)
        else:
            k = self.key(x)  # (B, T, C)
            q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        self.var = wei.var().detach()
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        self.wei = copy.deepcopy(wei.detach())
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
