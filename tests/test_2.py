import torch
from testgpt import HeadVer1, GPTVer2
from .conftest import train, generate, config, seed_everything


def test_gpt_v2_and_head_v1_generates_text_given_a_context():
    """
    implement: TODO 2 - 2
    """
    seed_everything(1337)
    head = HeadVer1()
    V, T, C = config['vocab_size'], config['block_size'], config['embed_size']
    lm = GPTVer2(head, V, T, C)
    train(lm)  # may take a while
    was = generate(lm, "The ", 30)
    expected = "The oo rmnt oedi srnvhe\nd oy  phou"
    assert expected == was


def test_head_v1_takes_an_average_of_the_past_into_account():
    """
    implement: TODO 2 - 1
    """
    x = torch.Tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    expected = torch.Tensor([[[1,   2,   3  ],
                              [2.5, 3.5, 4.5],
                              [4,   5,   6  ]]])
    head = HeadVer1()
    was = head(x)
    assert torch.allclose(expected, was)