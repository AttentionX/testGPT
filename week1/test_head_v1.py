import torch
from week1.src.modeling_head_v1 import HeadVer1


# --- testing v1 --- #
def test_head_v1_takes_an_average_of_the_past_into_account():
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    expected = torch.Tensor([[[1,   2,   3  ],
                              [2.5, 3.5, 4.5],
                              [4,   5,   6  ]]])
    head = HeadVer1()
    was = head(x)
    assert torch.allclose(expected, was)
