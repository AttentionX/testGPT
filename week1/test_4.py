import torch
from src import BigramLMVer2, HeadVer1


class DummyHead(HeadVer1):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_bigram_lm_v2_logits_order_is_preserved():
    x1 = torch.IntTensor([[1, 2, 6]])  # (B, T)
    x2 = torch.IntTensor([[2, 1, 6]])  # (B, T)
    V = 32
    C = 512
    lm = BigramLMVer2(DummyHead(), V, C)
    logits_1 = lm.logits(x1)  # (B, T) -> (B, T, C)
    logits_2 = lm.logits(x2)  # (B, T) -> (B, T, C)
    assert not torch.allclose(logits_1[:, -1, :], logits_2[:, -1, :])

