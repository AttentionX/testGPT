
import torch
from .gpt_v2 import GPTVer2


class GPTVer3(GPTVer2):

    def logits(self, indices: torch.Tensor) -> torch.Tensor:
        """
        :param indices: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        B, T = indices.shape
        C = self.token_embedding_table.weight.shape[1]
        # --- TODO 6 - 1 --- #
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(indices)  # (B, T) ->  (B, T, C)
        x = tok_emb + self.pos_encodings_v4(T, C).to(tok_emb.device)  # (B, T, C). broadcast add (T, C) across B.
        x = self.contextualizer(x)  # (B, T, C) ->  (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) @ (B, T, |V|) -> (B, T, |V|)
        # ----------------- #
        return logits

    @staticmethod
    def pos_encodings_v1(block_size: int, embed_size: int) -> torch.FloatTensor:
        """
        dead-simple positional encoding.

        """
        # --- TODO 6 - 2 --- #
        positions = torch.arange(block_size).unsqueeze(1).repeat(1, embed_size)  # (L) -> (L, 1) -> (L, C)
        return positions.float()  # (L, C)
        # ------------------ #

    @staticmethod
    def pos_encodings_v2(block_size: int, embed_size: int) -> torch.FloatTensor:
        """
        a normalized version of v1.
        but... time-delta is not constant across variable-length sentences.
        """
        # --- TODO 6 - 3 --- #
        positions = GPTVer3.pos_encodings_v1(block_size, embed_size)  # (L, C)
        positions = positions / block_size  # (L, C) -- normalize --> (L, C)
        return positions
        # ------------------ #

    @staticmethod
    def pos_encodings_v3(block_size: int, embed_size: int) -> torch.Tensor:
        """
        PE(w_pos) =
        """
        # --- TODO 6 - 4 --- #
        positions = torch.arange(block_size).view(-1, 1)  # -> (L)
        freqs = 0.0001 ** (torch.arange(embed_size)[::2] / embed_size).view(1, -1)  # (,) -> (H)
        encodings = torch.sin(freqs * positions)  # (L, H)
        return encodings
        # ------------------ #

    @staticmethod
    def pos_encodings_v4(block_size: int, embed_size: int) -> torch.Tensor:
        """
        :param block_size: length of the sequence (T)
        :param embed_size: number of embedding dimensions (C)
        :return: (L, C)
        """
        # --- TODO 6 - 5 --- #
        positions = torch.arange(block_size).view(-1, 1)  # -> (L)
        freqs = 0.0001 ** (torch.arange(embed_size)[::2] / embed_size).view(1, -1)  # (,) -> (C)
        encodings = torch.zeros(size=(block_size, embed_size))  # (L, C)
        encodings[:, ::2] = torch.sin(freqs * positions)  # evens = sin
        encodings[:, 1::2] = torch.cos(freqs * positions)  # odds = cos
        return encodings
        # ------------------ #

