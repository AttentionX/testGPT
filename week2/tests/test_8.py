"""
Running question: why do we need Feedforward?
"""
import torch
from ..src.block_v1_ffn import FeedForward
from ..src.block_v1 import BlockVer1
from ..src.block_v2 import BlockVer2
from ..src.multi_head_v2 import MultiHeadVer2
from ..src.gpt_v4 import GPTVer4
from .conftest import config, train


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


def test_ffn_helps():
    torch.manual_seed(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- multi-head --- #
    gpt = GPTVer4(MultiHeadVer2(T, C, n_heads), config['vocab_size'], C, T)
    losses_1 = train(gpt)
    # --- multi-head + ffn --- #
    gpt = GPTVer4(BlockVer1(MultiHeadVer2(T, C, n_heads), C), config['vocab_size'], C, T)
    losses_2 = train(gpt)
    # gpt should perform better with multi-head
    assert losses_1['train'] > losses_2['train']
    assert losses_1['val'] > losses_2['val']


def test_residual_conn_helps():
    torch.manual_seed(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- layers of multi-head + ffn --- #
    contextualizer = torch.nn.Sequential(*[BlockVer1(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], C, T)
    losses_1 = train(gpt)
    # --- layers of  multi-head + ffn + residual --- #
    contextualizer = torch.nn.Sequential(*[BlockVer2(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], C, T)
    losses_2 = train(gpt)
    # gpt should perform better with multi-head
    assert losses_1['train'] > losses_2['train']
    assert losses_1['val'] > losses_2['val']
