import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:
        z = x @ w + b
        y_hat=self.sigmoid(z)
        loss=0.5*(y_hat-y_true)**2
        dl_dy=y_hat-y_true # loss derivative by output
        dy_dz=y_hat*(1-y_hat) #sigmoid derivative
        dl_dz=dl_dy*dy_dz #chain: output*activation fn
        dz_dw=x # (wx)`x =x
        dl_dw=dl_dz*dz_dw # chain: prev layer-> weigths layer
        dl_db=dl_dz # bias derivative (c)` + fn(x)= 0 + fn(x)
        return (np.round(dl_dw,5),np.round(dl_db,5))
