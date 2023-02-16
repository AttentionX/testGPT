"""
- sinusoidal positional encoding
- learned positions
- measure the difference?
"""
import torch
from .gpt_v3 import GPTVer3


class GPTVer4(GPTVer3):
    """
    `GPTVer3`에서 positional encoding대신 position embedding으로 구현한 모델로 position 정보를 embedding table을 학습하여 얻는 모델입니다.
    """
    def __init__(self, contextualizer: torch.nn.Module, vocab_size: int, block_size: int, embed_size: int):
        super().__init__(contextualizer, vocab_size, block_size, embed_size)
        self.token_embedding_table = torch.nn.Embedding(vocab_size, embed_size)  # (|V|, C)
        self.pos_embedding_table = torch.nn.Embedding(block_size, embed_size)  # (T, C)
        self.ln_f = torch.nn.LayerNorm(embed_size)  # final layer norm

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T) ->  (B, T, C)
        pos_emb = self.pos_embedding_table(torch.arange(T).to(tok_emb.device))  # (T) -> (T, C)
        x = tok_emb + pos_emb  # broadcast-add (T, C) to (B, T, C) across B.
        x = self.contextualizer(x)  # (B, T, C) ->  (B, T, C)
        x = self.ln_f(x)  # layer norm
        logits = self.lm_head(x)  # (B, T, C) @ (B, T, |V|) -> (B, T, |V|)
        return logits
