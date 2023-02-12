"""
stacking blocks (block = MHA + FFN)
- residual connection
- test for
    - loss decreases (perplexity)
    - preserves positional encodings at the end of layers
"""
import torch


class GPTVer5(torch.nn.Module):
    pass
