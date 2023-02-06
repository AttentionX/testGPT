import torch
from src import BigramLMVer2, HeadVer1
from test_utils import train, generate


def test_bigram_lm_v2_logits_order_is_preserved():
    x1 = torch.IntTensor([[1, 2, 6]])  # (B, T)
    x2 = torch.IntTensor([[2, 1, 6]])  # (B, T)
    V = 32
    C = 512
    lm = BigramLMVer2(HeadVer1(), V, C)
    logits_1 = lm.logits(x1)  # (B, T) -> (B, T, C)
    logits_2 = lm.logits(x2)  # (B, T) -> (B, T, C)
    assert not torch.allclose(logits_1[:, -1, :], logits_2[:, -1, :])


def test_bigram_lm_v2_generates_text_given_a_context():
    torch.manual_seed(1337)
    lm = train(head_ver=1, lm_ver=2)  # may take a while
    expected = "The quick brown fox jumps over the lazy apour.\nDWi'sthaun.\nICHone,\nIU"
    was = generate(lm, "The quick brown fox jumps over the lazy", 30)
    assert expected == was
