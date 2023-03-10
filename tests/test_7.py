"""
Running question: why do we need more than one head?
"""
import timeit
import torch
from testgpt.multi_head_v1 import MultiHeadVer1
from testgpt.multi_head_v2 import MultiHeadVer2
from testgpt.head_v4 import HeadVer4
from testgpt.gpt_v4 import GPTVer4
from .conftest import config, train, seed_everything


def test_multi_head_helps():
    """
    But multi-head leads to faster convergence than single head.
    """
    seed_everything(1337)
    V, T, C, n_heads = config['vocab_size'], config['block_size'], config['embed_size'], config['n_heads']
    # --- HeadVer4: single-head --- #
    contextualizer = HeadVer4(T, C, C)
    gpt = GPTVer4(contextualizer, V, T, C)
    losses_1 = train(gpt)
    # --- MultiHeadVer4: multi-head --- #
    contextualizer = MultiHeadVer1(T, C, n_heads)
    gpt = GPTVer4(contextualizer, V, T, C)
    losses_multi = train(gpt)
    # gpt should converge faster with multi-head
    assert losses_1['val'] > losses_multi['val']


def test_head_ver_4_and_multi_head_ver_1_are_equally_expensive():
    """
    (Vaswani et al. 2017, Attention is all you need)
    "Due to the reduced dimension of each head,
    the total computational cost is similar to that of single-head attention with full dimensionality."
    dk = dv = dmodel/h, where h = number of heads.
    trainable parameters of multi-head ver 1 and head ver 4 must be the same because
    head_size = embed_size // n_heads
    """
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    head_v4 = HeadVer4(T, C, C)
    assert sum([p.numel() for p in multi_head_v1.heads.parameters() if p.requires_grad]) \
           == sum([p.numel() for p in head_v4.parameters() if p.requires_grad])


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


def test_multi_head_ver_1_and_multi_head_ver_2_are_logically_identical():
    """
    And they are logically identical.
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


