from typing import Union
import torch
from torch.nn import functional as F
from .gpt_v1 import GPTVer1
from .head_v1 import HeadVer1


class GPTVer2(GPTVer1):

    def __init__(self, contextualizer: Union[HeadVer1, torch.nn.Module],
                 vocab_size: int, block_size: int, embed_size: int):
        super().__init__(vocab_size, block_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.contextualizer = contextualizer
        self.token_embedding_table = torch.nn.Embedding(vocab_size, embed_size)  # (|V|, C)
        self.lm_head = torch.nn.Linear(embed_size, vocab_size)  # (C, |V|)

    def logits(self, indices: torch.Tensor) -> torch.Tensor:
        """
        :param indices: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        # --- TODO 2 --- #
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(indices)  # (B, T) ->  (B, T, C)
        x = self.contextualizer(tok_emb)  # (B, T, C) ->  (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) @ (B, T, |V|) -> (B, T, |V|)
        # -------------- #
        return logits

    @torch.no_grad()
    def generate(self, indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # --- TODO 2 - 2 --- #
            # get the predictions
            logits, _ = self(indices[:, -self.block_size:])  # (B, T + new) -> (B, T) ->  (B, T, C)
            # focus only on the last time step -> predict what comes next
            logits = logits[:, -1, :]  # (B, T, C) -> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C) # softmax
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) -> next idx sampling
            # append sampled index to the running sequence
            indices = torch.cat((indices, idx_next), dim=1)  # (B, T + 1)
            # ------------------ #
        return indices
