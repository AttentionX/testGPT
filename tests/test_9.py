"""
running question: why do we need LayerNorm?
"""
import torch
from testgpt.gpt_v4 import GPTVer4
from testgpt.layer_normalization import LayerNorm
from testgpt.block_v2 import BlockVer2
from testgpt.block_v3 import BlockVer3
from testgpt.multi_head_v2 import MultiHeadVer2
from .conftest import config, train, seed_everything


# test: gpt v4 learns faster with LayerNorm
def test_layer_norm_helps_when_network_is_deep():
    """
    layer norm helps because it mitigates vanishing gradient.
    """
    seed_everything(1337)
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
    assert losses_1['val'] > losses_2['val']


def test_layer_norm_mitigates_vanishing_gradient():
    """
    a simple experiment to see if layer norm mitigates vanishing gradient.
    """
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


def test_layer_norm_properly_normalizes_the_feature_dimension():
    """
    Layenorm mitigates vanishing gradient by normalizing the features dimension.
    """
    B, T, C = 32, 64, 512
    ln = LayerNorm(C)
    x = torch.randn(T, C)
    out = ln(x)
    mean_across_features = torch.round(out.mean(dim=-1))
    var_across_features = torch.round(out.var(dim=-1))
    assert torch.allclose(mean_across_features, torch.zeros(mean_across_features.shape))
    assert torch.allclose(var_across_features, torch.ones(var_across_features.shape))
