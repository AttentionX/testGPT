
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
        encodings = torch.arange(block_size).unsqueeze(1).repeat(1, embed_size)  # (L) -> (L, 1) -> (L, C)
        return encodings.float()  # (L, C)
        # ------------------ #

    @staticmethod
    def pos_encodings_v2(block_size: int, embed_size: int) -> torch.FloatTensor:
        """
        a normalized version of v1.
        but... time-delta is not constant across variable-length sentences.
        """
        # --- TODO 6 - 3 --- #
        encodings = GPTVer3.pos_encodings_v1(block_size, embed_size)  # (L, C)
        encodings = encodings / block_size  # (L, C) -- normalize --> (L, C)
        return encodings
        # ------------------ #

    @staticmethod
    def pos_encodings_v3(block_size: int, embed_size: int) -> torch.Tensor:
        """
        PE(pos, i) = sin(freq(i) * pos)
        where:
        freq(i) = 1/10000**(i/d_model)
        """
        # --- TODO 6 - 4 --- #
        # freq(i) = 1/10000**(i/d_model)
        freqs = 0.0001 ** (torch.arange(embed_size) / embed_size).unsqueeze(0)  # (C) -> (1, C)
        # ---  PE(pos, i) = sin(freq(i) * pos) --- #
        positions = torch.arange(block_size).unsqueeze(1)  # (L) -> (L, 1)
        encodings = torch.sin(positions * freqs)  # (L, 1) * (1, C) -> (L, C); multiply L across C
        return encodings
        # ------------------ #

    @staticmethod
    def pos_encodings_v4(block_size: int, embed_size: int) -> torch.Tensor:
        """
        PE(pos, 2i) = sin(freq(2i) * pos)
        PE(pos, 2i + 1) = cos(freq(2i) * pos)
        where:
        freq(2i) = 1/10000**(2i/d_model)
        """
        # --- TODO 6 - 5 --- #
        # freq(2i) = 1/10000**(2i/d_model)
        freqs = 0.0001 ** (torch.arange(embed_size)[::2] / embed_size).unsqueeze(0)  # (C) -> (1, C/2)
        # --- PE(pos, 2i) = sin(freq(2i) * pos) --- #
        # --- PE(pos, 2i + 1) = cos(freq(2i) * pos) --- #
        positions = torch.arange(block_size).unsqueeze(1)  # -> (L, 1)
        encodings = torch.zeros(size=(block_size, embed_size))  # (L, C)
        # evens = sin  (L, 1) * (1, C/2) ->  (L, C/2); multiply L across C/2
        encodings[:, ::2] = torch.sin(positions * freqs)
        # odds = cos  (L, 1) * (1, C/2) ->  (L, C/2); multiply L across C/2
        encodings[:, 1::2] = torch.cos(positions * freqs)
        return encodings
        # ------------------ #

