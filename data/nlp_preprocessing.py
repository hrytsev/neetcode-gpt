import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List
from torch.nn.utils.rnn import pad_sequence

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        words = {
        word
        for sentence in positive + negative
        for word in sentence.split()
        }

        words_sorted = sorted(list(words))
        words_table = {word: i + 1 for i, word in enumerate(words_sorted)}

        positive_encoded = [
        torch.tensor(
            [words_table[word] for word in sentence.split()],
            dtype=torch.float
        )
        for sentence in positive
                ]

        negative_encoded = [
        torch.tensor(
            [words_table[word] for word in sentence.split()],
            dtype=torch.float
        )
        for sentence in negative
        ]

        encoded = positive_encoded + negative_encoded

        return pad_sequence(encoded, batch_first=True, padding_value=0)