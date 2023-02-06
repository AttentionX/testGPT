import torch


class HeadVer2:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: out (B, T, C)
        """
        # ---- TODO --- #
        T = x.shape[1]
        wei = torch.tril(torch.ones(T, T)).to(x.device)
        wei = wei / wei.sum(1, keepdim=True)
        # perform the weighted aggregation of the values
        out = wei @ x  # (B, T, T) @ (B, T, C) -> (B, T, C)
        # ------------ #
        return out
