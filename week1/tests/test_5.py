import torch
from ..src import HeadVer4, GPTVer2
from .test_utils import config, train, generate


def test_head_v4_attention_has_no_notion_of_space():
    """
    :return:
    """
    x1 = torch.Tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
    x2 = torch.Tensor([[[4, 5, 6],
                        [1, 2, 3],
                        [7, 8, 9]]])
    _, T, C = x1.shape
    head = HeadVer4(T, C)
    y1 = head(x1)  # (B, T, C)
    y2 = head(x2)  # (B, T, C)
    assert torch.allclose(y1[:, -1, :], y2[:, -1, :])


def test_head_v4_logits_are_properly_masked():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    T, C = x.shape[1:]
    head = HeadVer4(T, C)
    head(x)
    expected = torch.IntTensor([[[0,  1,  1],
                                 [0,  0,  1],
                                 [0,  0,  0]]])
    # convert the Bool tensor to Int tensor
    was = (head.wei == 0.0).int()
    assert torch.allclose(expected, was)


def test_head_v4_logits_are_properly_normalized():
    B, T, C = 4, 10, 8
    x = torch.rand(B, T, C)
    head = HeadVer4(T, C)
    head(x)
    expected = torch.ones(B, T)
    was = head.wei.sum(dim=-1)
    assert torch.allclose(expected, was)


def test_head_v4_the_variance_of_wei_after_scale_is_1():
    B, T, C = 4, 128, 1024
    x = torch.randn(B, T, C)
    head = HeadVer4(T, C)
    head(x, test=True)  # (B, T, C)
    assert 1 == torch.round(head.var)


def test_gpt_v2_and_head_v4_generates_text_given_a_context():
    torch.manual_seed(1337)
    head = HeadVer4(config['block_size'], config['embed_size'])
    lm = GPTVer2(head, config['vocab_size'], config['embed_size'], config['block_size'])
    train(lm)  # may take a while
    expected = "The quick brown fox jumps over the lazyor th manot utou s l spaif ant"
    was = generate(lm, "The quick brown fox jumps over the lazy", 30)
    assert expected == was
