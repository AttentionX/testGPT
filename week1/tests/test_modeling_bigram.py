
from modeling_bigram import BigramLanguageModelV1, BigramLanguageModelV2  # noqa
import numpy as np

# --- testing v1 --- #


# --- testing v2 --- #
def test_bigram_language_model_v2_pos_encodings():
    """
    """
    T, C = 10, 1024
    encodings = BigramLanguageModelV2.pos_encodings(T, C)
    dist_1 = np.linalg.norm(encodings[2] - encodings[0])
    dist_2 = np.linalg.norm(encodings[3] - encodings[1])
    dist_3 = np.linalg.norm(encodings[4] - encodings[2])
    dist_4 = np.linalg.norm(encodings[5] - encodings[3])
    assert dist_1 == dist_2
    assert dist_2 == dist_3
    assert dist_3 == dist_4
