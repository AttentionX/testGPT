import numpy as np
from src import BigramLMVer2


def test_bigram_lm_v2_pos_encodings_dist_stays_constant():
    T, C = 10, 1024
    pos_encodings = BigramLMVer2.pos_encodings(T, C)
    dist_1 = np.linalg.norm(pos_encodings[2] - pos_encodings[0])
    dist_2 = np.linalg.norm(pos_encodings[3] - pos_encodings[1])
    dist_3 = np.linalg.norm(pos_encodings[4] - pos_encodings[2])
    dist_4 = np.linalg.norm(pos_encodings[5] - pos_encodings[3])
    assert dist_1 == dist_2
    assert dist_2 == dist_3
    assert dist_3 == dist_4

