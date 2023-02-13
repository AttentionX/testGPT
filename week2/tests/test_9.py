"""
running question: why do we need LayerNorm?
"""
from ..src.block_v3_ln import LayerNorm
import torch


def test_features_dim_is_properly_normalized():
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
    assert torch.allclose(torch.round(without_norm), torch.zeros(without_norm.shape))

    # Measure gradients with LayerNorm
    with_norm = x.clone().detach().requires_grad_(True)
    for i in range(depth):
        outputs = torch.nn.Linear(C, C)(with_norm)
        outputs = LayerNorm(C)(outputs)
        outputs.sum().backward()
        with_norm = outputs.clone().detach().requires_grad_(True)
    assert not torch.allclose(torch.round(with_norm), torch.zeros(with_norm.shape))


# test: gpt v4 learns faster with LayerNorm
def test_gpt_v4_learns_faster_with_layer_norm():
    pass
