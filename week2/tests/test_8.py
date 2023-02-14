"""
Running question: why do we need Feedforward?
"""
import torch
from ..src.block_v1_ffn import FeedForward


def test_ffn_is_applied_position_wise():
    """
    While the linear transformations are the same across different positions, they use different parameters from layer to layer.
    https://ai.stackexchange.com/q/15524
    """
    x_1 = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x_2 = torch.Tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
    T, C = x_1.shape
    ffn = FeedForward(C)
    y_1 = ffn(x_1)  # (T, C) ->  (T, C)
    y_2 = ffn(x_2)  # (T, C) ->  (T, C)
    assert torch.allclose(y_1[2, :], y_2[1, :])
    assert torch.allclose(y_1[0, :], y_2[2, :])
    assert torch.allclose(y_1[1, :], y_2[0, :])


# test: Feedforward is non-line
# test:
def test_gpt_v4_learns_better_with_feedforward():
    pass
