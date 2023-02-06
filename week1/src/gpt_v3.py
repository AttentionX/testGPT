import torch
from . import GPTVer2


class GPTVer3(GPTVer2):

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param idx: (B, T) tensor of integers
        :return: logits (B, T, |V|)
        """
        B, T = idx.shape
        C = self.token_embedding_table.weight.shape[1]
        # --- TODO 6 --- #
        logits = ...
        raise NotImplementedError
        # ------------- #
        return logits

    @staticmethod
    def pos_encodings(block_size: int, embed_size: int) -> torch.Tensor:
        """
        :param block_size: length of the sequence (T)
        :param embed_size: number of embedding dimensions (C)
        :return: (L, H)
        """
        # --- TODO 6 --- #
        encodings = ...
        raise NotImplementedError
        # -------------- #
        return encodings
