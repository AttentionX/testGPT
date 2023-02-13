import torch


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
