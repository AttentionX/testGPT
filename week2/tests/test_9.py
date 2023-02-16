"""
running question: why do we need LayerNorm?
"""
import torch
from ..src.gpt_v4 import GPTVer4
from ..src.layer_normalization import LayerNorm
from ..src.block_v2 import BlockVer2
from ..src.block_v3 import BlockVer3
from ..src.multi_head_v2 import MultiHeadVer2
from .conftest import config, train


def test_layer_norm_features_dim_is_properly_normalized():
    B, T, C = 32, 64, 512
    ln = LayerNorm(C)
    x = torch.randn(T, C)
    out = ln(x)
    mean_across_features = torch.round(out.mean(dim=-1))
    var_across_features = torch.round(out.var(dim=-1))
    assert torch.allclose(mean_across_features, torch.zeros(mean_across_features.shape))
    assert torch.allclose(var_across_features, torch.ones(var_across_features.shape))


def test_layer_norm_mitigates_vanishing_gradient():
    depth = 1000
    B, T, C = 3, 64, 128
    x = torch.randn(B, T, C, requires_grad=True)

    # Measure gradients without LayerNorm
    without_norm = x.clone().detach().requires_grad_(True)
    for i in range(depth):
        outputs = torch.nn.Linear(C, C)(without_norm)
        outputs.sum().backward()
        without_norm = outputs.clone().detach().requires_grad_(True)
    # gradients should be near-zero
    assert torch.allclose(torch.round(without_norm), torch.zeros(without_norm.shape))

    # Measure gradients with LayerNorm
    with_norm = x.clone().detach().requires_grad_(True)
    for i in range(depth):
        outputs = torch.nn.Linear(C, C)(with_norm)
        outputs = LayerNorm(C)(outputs)
        outputs.sum().backward()
        with_norm = outputs.clone().detach().requires_grad_(True)
    # gradients should not be near-zero
    assert not torch.allclose(torch.round(with_norm), torch.zeros(with_norm.shape))


# test: gpt v4 learns faster with LayerNorm
def test_layer_norm_helps_when_network_is_deep():
    torch.manual_seed(1337)
    T, C, n_heads = config['block_size'], config['embed_size'], config['n_heads']
    # --- layers of BlockVer2: multi-head + ffn + residual --- #
    contextualizer = torch.nn.Sequential(
        *[BlockVer2(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_1 = train(gpt)
    # --- layers of BlockVer3: multi-head + ffn + residual + layer norm --- #
    contextualizer = torch.nn.Sequential(
        *[BlockVer3(MultiHeadVer2(T, C, n_heads), C) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    losses_2 = train(gpt)
    assert losses_1['train'] > losses_2['train']
