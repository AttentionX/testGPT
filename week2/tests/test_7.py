"""
Running question: why do we need more than one head?
"""
import timeit
import torch
from ..src.multi_head_v1 import MultiHeadVer1
from ..src.multi_head_v2 import MultiHeadVer2
from ..src.multi_head_v1_head_v4 import HeadVer4
from ..src.gpt_v4 import GPTVer4
from .conftest import config, train


# test: multi-head ver 2 is logically the same as multi-head ver 1
def test_multi_head_ver_2_is_logically_the_same_as_multi_head_ver_1():
    B, T, C = 32, 64, 512
    n_heads = 8
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    multi_head_v2 = MultiHeadVer2(T, C, n_heads)
    # --- head v1 --- #
    for head in multi_head_v1.heads:
        # initialize weights with all zeros
        head.query.weight = torch.nn.Parameter(torch.zeros_like(head.query.weight))
        head.key.weight = torch.nn.Parameter(torch.zeros_like(head.key.weight))
        head.value.weight = torch.nn.Parameter(torch.zeros_like(head.value.weight))
    # --- head v2 --- #
    multi_head_v2.query.weight = torch.nn.Parameter(torch.zeros_like(multi_head_v2.query.weight))
    multi_head_v2.key.weight = torch.nn.Parameter(torch.zeros_like(multi_head_v2.key.weight))
    multi_head_v2.value.weight = torch.nn.Parameter(torch.zeros_like(multi_head_v2.value.weight))
    multi_head_v2.proj = multi_head_v1.proj
    x = torch.randn(B, T, C)
    out_v1 = multi_head_v1(x)
    out_v2 = multi_head_v2(x)
    assert torch.allclose(out_v1, out_v2)


# test: multi-head ver 2 is faster than ver 1
def test_multi_head_ver_2_is_faster_than_ver_1():
    B, T, C = 32, 64, 512
    n_heads = 8
    x = torch.randn(B, T, C)
    multi_head_v1 = MultiHeadVer1(T, C, n_heads)
    multi_head_v2 = MultiHeadVer2(T, C, n_heads)
    time_taken_v1 = timeit.timeit(lambda: multi_head_v1(x), number=10)
    time_taken_v2 = timeit.timeit(lambda: multi_head_v2(x), number=10)
    assert time_taken_v2 < time_taken_v1


# test: gpt learns faster with multi-head
def test_having_multiple_heads_helps():
    torch.manual_seed(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- single-head --- #
    contextualizer = torch.nn.Sequential(*[HeadVer4(T, C, C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], C, T)
    losses_single = train(gpt)
    # --- multi-head --- #
    contextualizer = torch.nn.Sequential(*[MultiHeadVer2(T, C, n_heads) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], C, T)
    losses_multi = train(gpt)
    # gpt should perform better with multi-head
    assert losses_single['train'] > losses_multi['train']
    assert losses_single['val'] > losses_multi['val']



