"""
implement  BigramLMVer1.logits()  method
and run this test with `pytest test_1.py -s --verbose`
"""
import torch
from src import BigramLMVer1


def test_bigram_lm_v1_logits_order_is_not_preserved():
    x1 = torch.IntTensor([[1, 2, 6]])  # (B, T)
    x2 = torch.IntTensor([[2, 1, 6]])  # (B, T)
    V = 32
    model = BigramLMVer1(V)
    logits_1 = model.logits(x1)  # (B, T) -> (B, T, C)
    logits_2 = model.logits(x2)  # (B, T) -> (B, T, C)
    assert torch.allclose(logits_1[:, -1, :], logits_2[:, -1, :])