"""
- sinusoidal positional encoding
- learned positions
- measure the difference?
"""
import torch
from torch.nn import functional as F
from typing import Optional, Union
from .block_v1 import BlockVer1
from .block_v2 import BlockVer2
from .block_v3 import BlockVer3
from .block_v4 import BlockVer4


class GPTVer4(torch.nn.Module):
    """
    learn positional embeddings from data
    """
    def __init__(self, block: Union[BlockVer1, BlockVer2, BlockVer3, BlockVer4],
                 vocab_size: int, embed_size: int, block_size: int):
        super().__init__()
        self.block = block
        self.token_embedding_table = torch.nn.Embedding(vocab_size, embed_size)  # (|V|, C)
        self.pos_embedding_table = torch.nn.Embedding(block_size, embed_size)  # (T, C)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) \
            -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param idx: (B, T) tensor of integers
        :param targets: (B, T) tensor of integers
        :return: logits (B, T, |V|), loss (scalar)
        """
        # idx and targets are both (B, T) tensor of integers
        logits = self.logits(idx)  # (B, T) ->  (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B, T, C) ->  (B * T, C)
            targets = targets.view(B * T)  # (B, T) -> (B * T)
            loss = F.cross_entropy(logits, targets)  # (B * T, C), (B * T) -> scalar
        return logits, loss

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        B, T = idx.shape
        # --- TODO --- #
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T) ->  (B, T, C)
        pos_emb = self.pos_embedding_table(torch.arange(T).to(tok_emb.device))  # (T) -> (T, C)
        x = tok_emb + pos_emb  # broadcast-add (T, C) to (B, T, C) across B.
        x = self.block(x)  # (B, T, C) ->  (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) @ (B, T, |V|) -> (B, T, |V|)
        # ----------- #
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :param max_new_tokens: maximum number of tokens to generate
        :return: (B, T + max_new_tokens) tensor of integers
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            last_idx = idx[:, -self.block_size:]  # (B, T + new) -> (B, T)
            logits, _ = self(last_idx)  # just get the logits
            # focus only on the last time step -> predict what comes next
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C) # softmax
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) -> next idx sampling
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
