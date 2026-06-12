import torch
import torch.nn as nn
import torch.nn.functional as F

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int,
              context_length: int, batch_size: int, lr: float) -> float:

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            ix = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            x = torch.stack([
                data[i:i + context_length]
                for i in ix
            ])

            y = torch.stack([
                data[i + 1:i + context_length + 1]
                for i in ix
            ])

            logits = model(x)  # [B, T, vocab_size]

            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.view(B * T, V),
                y.view(B * T)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # FIX 5: Python float rounding (не torch.round)
        return round(loss.item(), 4)