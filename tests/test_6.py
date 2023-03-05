"""
check if ver_1, ver_2, ver_3 preserves order.
"""
from typing import Callable

import pytest
import torch
import editdistance
from .conftest import config, train, generate, seed_everything
from testgpt import HeadVer1, HeadVer4, GPTVer1, GPTVer2, GPTVer3


def test_gpt_v1_and_v2_logits_order_is_not_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer1(V, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    model = GPTVer2(HeadVer1(), V, T, C)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])


def test_gpt_v3_pos_encodings_v1():
    """
    PE(w_pos) = w_pos
    """
    T, C = 4, 512
    # the first version of PE satisfies this property
    encodings = GPTVer3.pos_encodings_v1(T, C)
    short_encodings = GPTVer3.pos_encodings_v1(50, C)
    long_encodings = GPTVer3.pos_encodings_v1(100, C)
    # --- property 1 --- #
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- # (THIS DOES NOT HOLD)
    with pytest.raises(AssertionError):
        assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                          torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- #
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_pos_encodings_v2():
    """
    PE(w_pos) - w_pos / length
    """
    T, C = 4, 512
    # the first version of PE satisfies this property
    encodings = GPTVer3.pos_encodings_v2(T, C)
    short_encodings = GPTVer3.pos_encodings_v2(50, C)
    long_encodings = GPTVer3.pos_encodings_v2(100, C)
    # --- property 1 --- #
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- #
    assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    with pytest.raises(AssertionError):  # (THIS DOES NOT HOLD)
        assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                              torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- #
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_pos_encodings_v3():
    """
    PE(w_pos) = sin(w_pos / 10000^(i/C))
    """
    T, C = 4, 512
    # the first version of PE satisfies this property
    encodings = GPTVer3.pos_encodings_v3(T, C)
    short_encodings = GPTVer3.pos_encodings_v3(50, C)
    long_encodings = GPTVer3.pos_encodings_v3(100, C)
    # --- property 1 --- #
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- #
    assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                          torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- # (THIS DOES NOT HOLD)
    with pytest.raises(AssertionError):
        assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_pos_encodings_v4():
    """
    sinusoidal version of position encoding.
    PE(w_pos) = sin(w_pos / 10000^(i/C)) if i is even
    PE(w_pos) = cos(w_pos / 10000^((i)/C)) if i is odd
    should satisfy all properties
    """
    T, C = 4, 512
    # the first version of PE satisfies this property
    encodings = GPTVer3.pos_encodings_v4(T, C)
    short_encodings = GPTVer3.pos_encodings_v4(50, C)
    long_encodings = GPTVer3.pos_encodings_v4(100, C)
    # --- property 1 --- #
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    # --- property 2 --- #
    assert torch.all(torch.abs(encodings) <= 1)
    # --- property 3 --- #
    assert torch.allclose(torch.norm(short_encodings[2] - short_encodings[0]),
                          torch.norm(long_encodings[2] - long_encodings[0]))
    # --- property 4 --- #
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))


def test_gpt_v3_logits_order_is_preserved():
    """
    As opposed to GPTVer1 & GPTVer2,  GPTVer3 preserves the order of the input tokens.
    """
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer3(HeadVer1(), V, T, C)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert not torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert not torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert not torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v3_and_head_v4_generates_text_given_a_context():
    """
    With much to positional info, GPTVer3 picks up the Shakespearean pause.
    # --- from input.txt --- #
    e.g. We are accounted poor citizens, the patricians good.
    e.g. Let us kill him, and we'll have corn at our own price.
    e.g. I say unto you, what he hath done famously, he did
    """
    seed_everything(1337)
    V, T, C = config['vocab_size'], config['embed_size'], config['block_size']
    head = HeadVer4(T, C, C)
    lm = GPTVer3(head, V, T, C)
    train(lm)  # may take a while
    was = generate(lm, "The ", 30)
    assert editdistance.eval("The t weou fedothtotoutho,\nI- Iowh", was) < 5
