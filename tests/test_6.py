"""
check if ver_1, ver_2, ver_3 preserves order.
"""
import torch
from .conftest import config, train, generate, seed_everything
from testgpt import HeadVer1, HeadVer4, GPTVer1, GPTVer2, GPTVer3


def test_gpt_v3_and_head_v4_generates_text_given_a_context():
    """
    with positional encodings added, gpt picks up Shakespearean pause (comma), so to speak.
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
    expected = "The t weou fedothtotoutho,\nI- Iowh"
    assert expected == was


def test_gpt_v1_logits_order_is_not_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    model = GPTVer1(V, T)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v2_logits_order_is_not_preserved():
    torch.manual_seed(1337)
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer2(HeadVer1(), V, T, C)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert torch.allclose(logits[:, 2, :], logits[:, 3, :])


def test_gpt_v3_pos_encodings_each_pos_is_different():
    T, C = 4, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])
    assert not torch.allclose(encodings[2], encodings[3])


def test_gpt_v3_pos_encodings_dist_stays_constant():
    T, C = 10, 512
    encodings = GPTVer3.pos_encodings(T, C)
    assert torch.allclose(torch.norm(encodings[2] - encodings[0]), torch.norm(encodings[3] - encodings[1]))
    assert torch.allclose(torch.norm(encodings[5] - encodings[3]), torch.norm(encodings[6] - encodings[4]))
    assert torch.allclose(torch.norm(encodings[7] - encodings[5]), torch.norm(encodings[8] - encodings[6]))


def test_gpt_v3_logits_order_is_preserved():
    x = torch.IntTensor([[7, 7, 7, 7]])  # (B, T)
    _, T = x.shape
    V = 32
    C = 512
    model = GPTVer3(HeadVer1(), V, T, C)
    logits = model.logits(x)  # (B, T) -> (B, T, |V|)
    assert not torch.allclose(logits[:, 0, :], logits[:, 1, :])
    assert not torch.allclose(logits[:, 1, :], logits[:, 2, :])
    assert not torch.allclose(logits[:, 2, :], logits[:, 3, :])

