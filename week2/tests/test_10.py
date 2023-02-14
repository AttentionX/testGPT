"""
running question: why do we need Dropout?
"""
import pytest
import torch
from ..src.block_v4 import BlockVer4
from ..src.block_v3 import BlockVer3
from ..src.multi_head_v2 import MultiHeadVer2
from ..src.multi_head_v1 import MultiHeadVer1
from ..src.gpt_v4 import GPTVer4
from .conftest import config, train, generate


def test_block_ver_4_output_is_always_different_in_train_mode():
    B, T, C = 32, 64, 512
    n_heads = 8
    dropout = 0.1
    block = BlockVer4(MultiHeadVer1(T, C, n_heads), C, dropout)
    block.train()
    x = torch.randn(B, T, C)
    out_1 = block(x)
    out_2 = block(x)
    out_3 = block(x)
    out_4 = block(x)
    assert not torch.allclose(out_1, out_2)
    assert not torch.allclose(out_2, out_3)
    assert not torch.allclose(out_3, out_4)


def test_block_ver_4_output_is_always_the_same_in_eval_mode():
    B, T, C = 32, 64, 512
    n_heads = 8
    dropout = 0.1
    block = BlockVer4(MultiHeadVer1(T, C, n_heads), C, dropout)
    block.eval()
    x = torch.randn(B, T, C)
    out_1 = block(x)
    out_2 = block(x)
    out_3 = block(x)
    out_4 = block(x)
    assert torch.allclose(out_1, out_2)
    assert torch.allclose(out_2, out_3)
    assert torch.allclose(out_3, out_4)


def test_dropout_helps():
    """
    dropout mitigates overfitting
    """
    torch.manual_seed(1337)
    T, C, n_heads, dropout = config['block_size'], config['embed_size'], config['n_heads'], config['dropout']
    # --- layers of multi-head + ffn + residual + layer norm --- #
    contextualizer = torch.nn.Sequential(
        *[BlockVer3(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- layers of  multi-head + ffn + residual + layer norm + dropout --- #
    contextualizer = torch.nn.Sequential(
        *[BlockVer4(MultiHeadVer2(T, C, n_heads), C, dropout) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    assert losses_1['train'] < losses_2['train']
    assert losses_1['val'] > losses_2['val']


def test_generate_with_gpt():
    torch.manual_seed(1337)
    T, C, n_heads, dropout = config['block_size'], config['embed_size'], config['n_heads'], config['dropout']
    contextualizer = torch.nn.Sequential(
        *[BlockVer4(MultiHeadVer2(T, C, n_heads), C, dropout) for _ in range(config['n_layers'])])
    lm = GPTVer4(contextualizer, config['vocab_size'], T, C)
    train(lm)  # may take a while
    expected = """Aayyyayy sefd.\n\nCAUSCEN:\nEutpelun, wase wans and igoess that sIaes, hacony Garrerel,\nWot.\n\nARIIO ter """
    was = generate(lm, "A", 100)
    assert expected == was




