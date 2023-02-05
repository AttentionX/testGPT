import torch
from modeling_heads import HeadVer1, HeadVer2, HeadVer3, HeadVer4


# --- testing v1 --- #
def test_head_v1():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    answer = torch.Tensor([[[1,   2,   3  ],
                            [2.5, 3.5, 4.5],
                            [4,   5,   6  ]]])
    head = HeadVer1()
    assert torch.allclose(head(x), answer)


# --- testing v2 --- #
def test_head_v2():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    answer = torch.Tensor([[[1,   2,   3  ],
                            [2.5, 3.5, 4.5],
                            [4,   5,   6  ]]])
    head = HeadVer2()
    assert torch.allclose(head(x), answer)


# --- testing v3 --- #
def test_head_v3():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    answer = torch.Tensor([[[1,   2,   3  ],
                            [2.5, 3.5, 4.5],
                            [4,   5,   6  ]]])
    head = HeadVer3()
    assert torch.allclose(head(x), answer)


# --- testing v4 --- #
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


def test_head_v4_why_divide_by_sqrt_of_n_embd():
    B, T, C = 4, 128, 1024
    x = torch.randn(B, T, C)
    head = HeadVer4(T, C)
    head(x, debug=True)  # (B, T, C)
    assert 1 == torch.round(head.var)

