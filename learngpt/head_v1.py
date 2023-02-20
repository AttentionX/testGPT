import torch


class HeadVer1:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x (B, T, C)
        :return: out (B, T, C)
        """
        B, T, C = x.shape
        # --- TODO 2 --- #
        # use nested for loops
        out = torch.zeros((B, T, C)).to(x.device)
        for b in range(B):
            for t in range(T):
                xprev = x[b, :t + 1]
                out[b, t] = torch.mean(xprev, 0)
        # -------------- #
        return out
