"""
Running question: why do we need Feedforward?
"""
import torch
from ..src.block_v1 import BlockVer1
from ..src.block_v2 import BlockVer2
from ..src.multi_head_v2 import MultiHeadVer2
from ..src.gpt_v4 import GPTVer4
from .conftest import config, train


def test_ffn_helps():
    torch.manual_seed(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- MultiHeadVer2: multi-head --- #
    contextualizer = MultiHeadVer2(T, C, n_heads)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- BlockVer1: multi-head + ffn --- #
    contextualizer = BlockVer1(MultiHeadVer2(T, C, n_heads), C)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    assert losses_1['train'] > losses_2['train']


def test_residual_conn_helps_when_network_is_deep():
    torch.manual_seed(1337)
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
    assert losses_1['train'] > losses_2['train']
