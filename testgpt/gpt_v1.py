from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTVer1(nn.Module):

    def __init__(self, vocab_size: int, block_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.block_size = block_size

    def forward(self, indices: torch.Tensor, targets: Optional[torch.Tensor] = None) -> \
            tuple[torch.Tensor, Optional[torch.Tensor]]:
        # idx and targets are both (B, T) tensor of integers
        logits = self.logits(indices)  # (B, T) ->  (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B, T, C) ->  (B * T, C)
            targets = targets.view(B * T)  # (B, T) -> (B * T)
            loss = F.cross_entropy(logits, targets)  # (B * T, C), (B * T) -> scalar
        return logits, loss

    def logits(self, indices: torch.Tensor) -> torch.Tensor:
        """
        GPT v2 uses token embeddings as the logits.
        """
        return self.token_embedding_table(indices)  # (B, T) ->  (B, T, C)

    @torch.no_grad()
    def generate(self, indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # --- TODO 1 --- #
            # get the predictions
            logits, _ = self(indices[:, -1])  # (B, T + new) -> (B, 1) ->  (B, 1, C)
            # focus only on the last time step -> predict what comes next
            logits = logits.squeeze(1)  # (B, 1, C) -> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C) --normalize over C-->  (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) -> next idx sampling
            # append sampled index to the running sequence
            indices = torch.cat((indices, idx_next), dim=1)  # (B, T + 1)
            # -------------- #
        return indices
