import torch
from modeling_bigram_lm_v2 import BigramLMVer2
from modeling_head_v1 import HeadVer1
import numpy as np


def test_bigram_lm_v2_pos_encodings_dist_stays_constant():
    T, C = 10, 1024
    encodings = BigramLMVer2.pos_encodings(T, C)
    dist_1 = np.linalg.norm(encodings[2] - encodings[0])
    dist_2 = np.linalg.norm(encodings[3] - encodings[1])
    dist_3 = np.linalg.norm(encodings[4] - encodings[2])
    dist_4 = np.linalg.norm(encodings[5] - encodings[3])
    assert dist_1 == dist_2
    assert dist_2 == dist_3
    assert dist_3 == dist_4


def test_bigram_lm_v2_logits_order_is_preserved():
    x1 = torch.IntTensor([[1, 2, 6]])  # (B, T)
    x2 = torch.IntTensor([[2, 1, 6]])  # (B, T)
    V = 32
    C = 512
    model = BigramLMVer2(HeadVer1(), V, C)
    logits_1 = model.logits(x1)  # (B, T) -> (B, T, C)
    logits_2 = model.logits(x2)  # (B, T) -> (B, T, C)
    assert not torch.allclose(logits_1[:, -1, :], logits_2[:, -1, :])