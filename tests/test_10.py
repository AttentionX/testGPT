"""
running question: why do we need Dropout?
"""
import torch
from testgpt.block_v4 import BlockVer4
from testgpt.block_v3 import BlockVer3
from testgpt.multi_head_v2 import MultiHeadVer2
from testgpt.multi_head_v1 import MultiHeadVer1
from testgpt.gpt_v4 import GPTVer4
from .conftest import config, train, seed_everything
from . import conftest


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
    dropout helps because it mitigates overfitting.
    """
    seed_everything(1337)
    T, C, n_heads, dropout = config['block_size'], config['embed_size'], config['n_heads'], config['dropout']
    #  --- push the model to overfit --- #
    train_ratio = 0.001
    n = int(len(conftest.data) * train_ratio)
    conftest.train_data = conftest.data[:n]
    conftest.val_data = conftest.data[n:]
    config['max_iters'] = 10000
    config['learning_rate'] = 0.005
    # --- BlockVer3: layers of multi-head + ffn + residual + layer norm --- #
    contextualizer = BlockVer3(MultiHeadVer2(T, C, n_heads), C)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- BlockVer4: layers of multi-head + ffn + residual + layer norm + dropout --- #
    contextualizer = BlockVer4(MultiHeadVer2(T, C, n_heads), C, dropout)
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    # "mitigates overfitting" = train loss is bigger but validation loss is smaller.
    assert losses_1['train'] < losses_2['train']
    assert losses_1['val'] > losses_2['val']


