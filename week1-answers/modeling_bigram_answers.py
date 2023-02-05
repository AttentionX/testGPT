from typing import Union
from modeling_heads_answers import HeadVer1, HeadVer2, HeadVer3, HeadVer4
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModelVer1(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.logits(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.token_embedding_table(idx)  # (B, T) ->  (B, T, C)

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        # idx is (B, T) array of indices in the current context
        _, T = idx.shape
        for _ in range(max_new_tokens):
            # get the predictions
            last_idx = idx[:, -T:]   # (B, T + new) -> (B, T)
            logits, loss = self(last_idx)  # embedding vector, loss=None
            # focus only on the last time step -> predict what comes next
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C) # softmax
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) -> next idx sampling
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class BigramLanguageModelVer2(BigramLanguageModelVer1):

    def __init__(self, vocab_size: int, head: Union[HeadVer1, HeadVer2, HeadVer3, HeadVer4], n_embd: int):
        super().__init__(vocab_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.head = head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        C = self.token_embedding_table.weight.shape[1]
        # --- TODO --- #
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T) -> (B,T,C)
        pos_emb = self.pos_encodings(T, C).to(idx.device)  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.head(x)  # apply one head  of self-attention. (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        # ------------ #
        return logits

    @staticmethod
    def pos_encodings(block_size: int, n_embd: int) -> torch.Tensor:
        """
        :return: (L, H)
        """
        # --- TODO --- #
        positions = torch.arange(block_size).view(-1, 1)  # -> (L)
        freqs = 0.0001 ** (torch.arange(n_embd)[::2] / n_embd).view(1, -1)  # (,) -> (H)
        encodings = torch.zeros(size=(block_size, n_embd))  # (L, H)
        encodings[:, ::2] = torch.sin(freqs * positions)  # evens = sin
        encodings[:, 1::2] = torch.cos(freqs * positions)
        return encodings
        # ------------ #
