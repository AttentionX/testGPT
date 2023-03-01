"""
Running question: why do we need Feedforward?
"""
import torch
from testgpt.block_v1 import BlockVer1
from testgpt.block_v2 import BlockVer2
from testgpt.multi_head_v2 import MultiHeadVer2
from testgpt.gpt_v4 import GPTVer4
from .conftest import config, train, seed_everything


def test_ffn_helps():
    seed_everything(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- MultiHeadVer2: multi-head --- #
    contextualizer = MultiHeadVer2(T, C, n_heads)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- BlockVer1: multi-head + ffn --- #
    contextualizer = BlockVer1(MultiHeadVer2(T, C, n_heads), C)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    assert losses_1['val'] > losses_2['val']


def test_residual_conn_helps_when_network_is_deep():
    seed_everything(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- Layers of BlockVer1: multi-head + ffn --- #
    contextualizer = torch.nn.Sequential(*[BlockVer1(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- Layers of BlockVer2: multi-head + ffn + residual --- #
    contextualizer = torch.nn.Sequential(*[BlockVer2(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    # gpt should perform better with multi-head
    assert losses_1['val'] > losses_2['val']
