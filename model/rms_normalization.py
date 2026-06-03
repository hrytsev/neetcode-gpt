import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x=np.array(x)
        n=len(x)
        rms_x=np.sqrt( 1/n*np.sum(x**2)+eps)
        x_hat=x/rms_x
        output=x_hat*gamma
        return np.round(output,4)
