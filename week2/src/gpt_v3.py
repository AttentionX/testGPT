import torch
from typing import Optional
from torch.nn import functional as F


class GPTVer3(torch.nn.Module):
    """
    redefining this here in case you haven't completed week1/src/gpt_v3.py
    Week1에서 구현했던 모델입니다.  Week2를 바로 시작할 수 있도록 재정의 했습니다.
    """

    def __init__(self, contextualizer: torch.nn.Module, vocab_size: int, block_size: int, embed_size: int):
        # each token directly reads off the logits for the next token from a lookup table
        super().__init__()
        self.contextualizer = contextualizer
        self.block_size = block_size
        self.token_embedding_table = torch.nn.Embedding(vocab_size, embed_size)  # (|V|, C)
        self.lm_head = torch.nn.Linear(embed_size, vocab_size)  # (C, |V|)

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
        C = self.token_embedding_table.weight.shape[1]
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T) ->  (B, T, C)
        x = tok_emb + self.pos_encodings(T, C).to(tok_emb.device)  # (B, T, C). broadcast add (T, C) across B.
        x = self.contextualizer(x)  # (B, T, C) ->  (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) @ (B, T, |V|) -> (B, T, |V|)
        return logits

    @staticmethod
    def pos_encodings(block_size: int, embed_size: int) -> torch.Tensor:
        """
        :param block_size: length of the sequence (T)
        :param embed_size: number of embedding dimensions (C)
        :return: (L, H)
        """
        positions = torch.arange(block_size).view(-1, 1)  # -> (L)
        freqs = 0.0001 ** (torch.arange(embed_size)[::2] / embed_size).view(1, -1)  # (,) -> (H)
        encodings = torch.zeros(size=(block_size, embed_size))  # (L, H)
        encodings[:, ::2] = torch.sin(freqs * positions)  # evens = sin
        encodings[:, 1::2] = torch.cos(freqs * positions)
        return encodings

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
