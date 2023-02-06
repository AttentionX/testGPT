import torch


class HeadVer2:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, C)
        :return: out (B, T, C)
        """
        # --- TODO 3 --- #
        # vectorize HeadVer1.__call__()
        out = ...
        raise NotImplementedError
        # ------------ #
        return out
