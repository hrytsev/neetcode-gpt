import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        self.heads=nn.ModuleList([self.SingleHeadAttention(embedding_dim,attention_dim // num_heads) for _ in range(num_heads)])
            
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        self.output=nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        mha_result=torch.Tensor()
        for head in self.heads:
            head_result=head(embedded)
            mha_result=torch.concat((mha_result,head_result),dim=2)
        feed_forwad=self.output(mha_result)
        return torch.round(feed_forwad,decimals=4)
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places
        

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
