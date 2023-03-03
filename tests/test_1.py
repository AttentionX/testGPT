"""
implement  BigramLMVer1.logits()  method
and run this test with `pytest test_1.py -s --verbose`
"""
import torch
from testgpt import GPTVer1
from .conftest import config, train, generate, seed_everything


def test_gpt_v1_generates_text_given_a_context():
    """
    Note how gpt v1 ends the sentence rather abruptly.
    """
    seed_everything(1337)
    lm = GPTVer1(config['vocab_size'], config['block_size'])
    train(lm)
    was = generate(lm, "The ", 30)
    expected = "The \nSXro sick's q-etcichors "
    assert expected == was
