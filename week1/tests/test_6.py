"""
check if ver_1, ver_2, ver_3 preserves order.
"""
import numpy as np
import torch
from ..src import HeadVer1, GPTVer1, GPTVer2, GPTVer3


def test_gpt_v1_logits_order_is_not_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    model = GPTVer1(V, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v2_logits_order_is_not_preserved():
    torch.manual_seed(1337)
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer2(HeadVer1(), V, C, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v3_pos_encodings_each_pos_is_different():
    T, C = 4, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    assert not torch.allclose(encodings[2], encodings[3])


def test_gpt_v3_pos_encodings_dist_stays_constant():
    T, C = 10, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert np.linalg.norm(encodings[2] - encodings[0]) == np.linalg.norm(encodings[3] - encodings[1])
    assert np.linalg.norm(encodings[4] - encodings[2]) == np.linalg.norm(encodings[5] - encodings[3])
    assert np.linalg.norm(encodings[6] - encodings[4]) == np.linalg.norm(encodings[7] - encodings[5])


def test_gpt_v3_logits_order_is_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer3(HeadVer1(), V, C, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert not torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert not torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert not torch.allclose(logits[:, 2, :], logits[:, 3, :])



