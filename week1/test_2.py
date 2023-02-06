"""
implement BigramLMVer1.generate() method
and run this test with `pytest test_2.py -s --verbose`
"""
import torch
from test_utils import train, generate


def test_bigram_lm_v1_generates_text_given_a_context():
    torch.manual_seed(1337)
    lm = train(1, 1)  # may take a while
    was = generate(lm, "The quick brown fox jumps over the lazy", 30)
    expected = "The quick brown fox jumps over the lazy apour.\nDWi'sthaun.\nICHone,\nIU"
    assert expected == was
