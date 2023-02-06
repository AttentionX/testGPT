import torch
import timeit
from ..src import HeadVer1, HeadVer3


def test_head_v3_logically_the_same_as_head_v1():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head_v1 = HeadVer1()
    head_v3 = HeadVer3()
    y_v1 = head_v1(x)
    y_v3 = head_v3(x)
    assert torch.allclose(y_v1, y_v3)


def test_head_v3_faster_than_head_v1():
    x = torch.rand(4, 128, 1024)
    head_v1 = HeadVer1()
    head_v3 = HeadVer3()
    time_taken_v1 = timeit.timeit(lambda: head_v1(x), number=10)
    time_taken_v3 = timeit.timeit(lambda: head_v3(x), number=10)
    assert time_taken_v3 < time_taken_v1


def test_head_v3_logits_are_properly_normalized():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head = HeadVer3()
    head(x)
    expected = torch.IntTensor([[[0, 1, 1],
                                 [0, 0, 1],
                                 [0, 0, 0]]])
    # convert the Bool tensor to Int tensor
    was = (head.wei == 0.0).int()
    assert torch.allclose(expected, was)


def test_head_v3_logits_are_properly_masked():
    B, T, C = 4, 10, 8
    x = torch.rand(B, T, C)
    head = HeadVer3()
    head(x)
    expected = torch.ones(B, T)
    was = head.wei.sum(dim=-1)
    assert torch.allclose(expected, was)
