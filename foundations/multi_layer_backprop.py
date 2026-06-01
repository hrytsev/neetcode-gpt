import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=np.float64).reshape(-1, 1)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64).reshape(-1, 1)

        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64).reshape(-1, 1)

        y_true = np.array(y_true, dtype=np.float64).reshape(-1, 1)
        z1 = W1 @ x + b1
        a1=np.maximum(0,z1)
        z2=W2 @ a1 + b2

        loss=np.mean((z2-y_true)**2)
        n=len(y_true)
        
        dl_dz2=2/n*(z2-y_true) # by loss
        
        dl_dw2=dl_dz2 @ a1.T # by W2
        dl_db2 = np.sum(dl_dz2, axis=1, keepdims=True)

            
        dl_da1=W2.T @ dl_dz2
        dl_dz1= dl_da1 * (z1>0)

        dl_dw1=dl_dz1@x.T
        dl_db1 = np.sum(dl_dz1, axis=1, keepdims=True)
        
        
        return {
    'loss': float(np.round(loss, 4)),
    'dW1': np.round(dl_dw1, 4).tolist(),
    'db1': np.round(dl_db1, 4).flatten().tolist(),
    'dW2': np.round(dl_dw2, 4).tolist(),
    'db2': np.round(dl_db2, 4).flatten().tolist()
}
