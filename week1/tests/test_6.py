"""
check if ver_1, ver_2, ver_3 preserves order.
"""
import numpy as np
import torch
from ..src import GPTVer3, HeadVer4


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


def test_gpt_v3_logits_order_is_preserved_with_head_v4():
    x1 = torch.IntTensor([[1, 2, 6]])  # (B, T)
    x2 = torch.IntTensor([[2, 1, 6]])  # (B, T)
    B, T = x1.shape
    V = 32
    C = 512
    model = GPTVer3(HeadVer4(T, C), V, C, T)
    y1, _ = model(x1)  # (B, T) -> (B, T, C)
    y2, _ = model(x2)  # (B, T) -> (B, T, C)
    assert not torch.allclose(y1[:, -1, :], y2[:, -1, :])


