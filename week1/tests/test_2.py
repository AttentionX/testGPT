import torch
from ..src import HeadVer1, GPTVer2
from .test_utils import train, generate, config


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


def test_gpt_v2_and_head_v1_generates_text_given_a_context():
    torch.manual_seed(1337)
    head = HeadVer1()
    lm = GPTVer2(head, config['vocab_size'], config['embed_size'], config['block_size'])
    train(lm)  # may take a while
    expected = "The quick brown fox jumps over the lazyvee\nd ont phour teo, nwch aydo"
    was = generate(lm, "The quick brown fox jumps over the lazy", 30)
    assert expected == was
