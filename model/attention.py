import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.query=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.value=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.attention_dim=attention_dim
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        k,q,v=self.key(embedded),self.query(embedded),self.value(embedded)
        scores=(  q @ k.transpose(-2, -1))/(self.attention_dim**0.5)
        masked=torch.tril(scores)
        masked[masked==0]=float("-inf")
        softmax=nn.Softmax(dim=2)
        probablilites=softmax(masked)
        result=probablilites @ v
        return torch.round(result,decimals=4)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        pass
