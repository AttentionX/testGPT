"""
Running question: why do we need more than one head?
"""
import timeit
import torch
from ..src.multi_head_v1 import MultiHeadVer1
from ..src.multi_head_v2 import MultiHeadVer2
from ..src.head_v4 import HeadVer4
from ..src.gpt_v4 import GPTVer4
from .conftest import config, train


def test_head_ver_4_and_multi_head_ver_1_are_equally_expensive():
    """
    trainable parameters of multi-head ver 1 and head ver 4 must be the same because
    head_size = embed_size // n_heads
    """
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    head_v4 = HeadVer4(T, C, C)
    assert sum([p.numel() for p in multi_head_v1.heads.parameters() if p.requires_grad]) \
           == sum([p.numel() for p in head_v4.parameters() if p.requires_grad])


def test_multi_head_helps():
    """
    But multi-head leads to faster convergence than single head.
    """
    torch.manual_seed(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- HeadVer4: single-head --- #
    contextualizer = HeadVer4(T, C, C)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- MultiHeadVer4: multi-head --- #
    contextualizer = MultiHeadVer1(T, C, n_heads)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_multi = train(gpt)
    # gpt should perform better with multi-head
    assert losses_1['train'] > losses_multi['train']


def test_multi_head_ver_2_is_faster_than_ver_1():
    """
    MultiHeadVer2 is faster than MultiHeadVer1 because it does not involve explicit loops.
    """
    B, T, C = 32, 64, 512
    n_heads = 8
    x = torch.randn(B, T, C)
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    multi_head_v2 = MultiHeadVer2(T, C, n_heads)
    time_taken_v1 = timeit.timeit(lambda: multi_head_v1(x), number=10)
    time_taken_v2 = timeit.timeit(lambda: multi_head_v2(x), number=10)
    assert time_taken_v2 < time_taken_v1


def test_multi_head_ver_1_and_multi_head_ver_2_are_logically_equal():
    """
    And they are logically equal.
    """
    B, T, C = 1, 3, 8
    n_heads = 4
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    multi_head_v2 = MultiHeadVer2(T, C, n_heads)
    multi_head_v2.query.weight = torch.nn.Parameter(torch.concat([head.query.weight
                                                                  for head in multi_head_v1.heads], dim=0))
    multi_head_v2.key.weight = torch.nn.Parameter(torch.concat([head.key.weight
                                                                for head in multi_head_v1.heads], dim=0))
    multi_head_v2.value.weight = torch.nn.Parameter(torch.concat([head.value.weight
                                                                  for head in multi_head_v1.heads], dim=0))
    multi_head_v2.proj.weight = torch.nn.Parameter(multi_head_v1.proj.weight)
    multi_head_v2.proj.bias = torch.nn.Parameter(multi_head_v1.proj.bias)
    x = torch.randn(B, T, C)
    out_1 = multi_head_v1(x)
    out_2 = multi_head_v2(x)
    assert torch.allclose(out_1, out_2)




