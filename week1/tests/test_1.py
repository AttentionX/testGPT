"""
implement  BigramLMVer1.logits()  method
and run this test with `pytest test_1.py -s --verbose`
"""
import torch
from ..src import GPTVer1
from .test_utils import config, train, generate


def test_gpt_v1_logits_order_is_not_preserved():
    x1 = torch.IntTensor([[1, 2, 6]])  # (B, T)
    x2 = torch.IntTensor([[2, 1, 6]])  # (B, T)
    _, T = x1.shape
    V = 100
    model = GPTVer1(V, T)
    logits_1 = model.logits(x1)  # (B, T) -> (B, T, C)
    logits_2 = model.logits(x2)  # (B, T) -> (B, T, C)
    assert torch.allclose(logits_1[:, -1, :], logits_2[:, -1, :])


def test_gpt_v1_generates_text_given_a_context():
    torch.manual_seed(1337)
    lm = GPTVer1(config['vocab_size'], config['block_size'])
    train(lm)
    was = generate(lm, "The quick brown fox jumps over the lazy", 30)
    expected = "The quick brown fox jumps over the lazy:\nHAGdirdo sick's q-Whe,\n\nANs "
    assert expected == was
