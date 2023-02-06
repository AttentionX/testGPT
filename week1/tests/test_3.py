import timeit
import torch
from ..src import HeadVer1, HeadVer2


# --- testing v2 --- #
def test_head_v2_logically_the_same_as_head_v2():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    head_v1 = HeadVer1()
    head_v2 = HeadVer2()
    y_v1 = head_v1(x)
    y_v2 = head_v2(x)
    assert torch.allclose(y_v1, y_v2)


def test_head_v2_faster_than_head_v1():
    x = torch.rand(4, 128, 1024)
    head_v1 = HeadVer1()
    head_v2 = HeadVer2()
    time_taken_v1 = timeit.timeit(lambda: head_v1(x), number=10)
    time_taken_v2 = timeit.timeit(lambda: head_v2(x), number=10)
    assert time_taken_v2 < time_taken_v1
