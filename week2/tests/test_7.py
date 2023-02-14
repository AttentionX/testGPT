"""
Running question: why do we need more than one head?
"""
import torch
from ..src.multi_head_v1 import MultiHeadVer1
import random

# test: different heads learn different things
def test_multi_head_ver_1_different_heads_learn_different_things():
    """
    각 헤드가 다르다만 테스트 하자.
    :return:
    """
    random.seed(1337)
    B, T, C = 32, 64, 512
    n_heads = 8
    multihead = MultiHeadVer1(T, C, n_heads)
    x = torch.randn(B, T, C)
    out = multihead(x).view([n_heads, -1])
    idx = [random.sample(range(n_heads), 2) for _ in range(3)]
    
    assert not torch.allclose(out[idx[0][0]], out[idx[0][1]])
    assert not torch.allclose(out[idx[1][0]], out[idx[1][1]])
    assert not torch.allclose(out[idx[2][0]], out[idx[2][1]])


# test: multi-head ver 2 is logically the same as multi-head ver 1
def test_multi_head_ver_2_is_logically_the_same_as_multi_head_ver_1():
    pass


# test: multi-head ver 2 is faster than ver 1
def test_multi_head_ver_2_is_faster_than_ver_1():
    pass


# test: gpt learns faster with multi-head
def test_gpt_v4_learns_better_with_multi_head():
    pass

