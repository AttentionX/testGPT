import torch
from modeling_bigram import BigramLanguageModelVer2
from modeling_heads import HeadVer4
import numpy as np


# --- testing v2 --- #
def test_bigram_language_model_v2_pos_encodings_dist_stays_constant():
    T, C = 10, 1024
    encodings = BigramLanguageModelVer2.pos_encodings(T, C)
    dist_1 = np.linalg.norm(encodings[2] - encodings[0])
    dist_2 = np.linalg.norm(encodings[3] - encodings[1])
    dist_3 = np.linalg.norm(encodings[4] - encodings[2])
    dist_4 = np.linalg.norm(encodings[5] - encodings[3])
    assert dist_1 == dist_2
    assert dist_2 == dist_3
    assert dist_3 == dist_4


def test_bigram_language_model_v2_order_is_preserved():

    x1 = torch.IntTensor([[1, 2, 6]])  # (B, T)
    x2 = torch.IntTensor([[2, 1, 6]])  # (B, T)
    B, T = x1.shape
    V = 32
    C = 512
    model = BigramLanguageModelVer2(V, HeadVer4(T, C), C)
    y1, _ = model(x1)  # (B, T) -> (B, T, C)
    y2, _ = model(x2)  # (B, T) -> (B, T, C)
    assert not torch.allclose(y1[:, -1, :], y2[:, -1, :])
