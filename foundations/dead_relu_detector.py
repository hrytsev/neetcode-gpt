import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        out=x
        dead=[]
        with torch.no_grad():
            for layer in model.children():
                out=layer(out)
                if isinstance(layer,nn.ReLU):
                    dead_neurons = (out == 0).all(dim=0)
                    dead_fraction = dead_neurons.float().mean().item()
                    dead.append(round(dead_fraction, 4))
        return dead


    def suggest_fix(self,dead_fractions):
    
    # 1. use_leaky_relu
        if any(f > 0.5 for f in dead_fractions):
            return "use_leaky_relu"

    # 2. reinitialize
        if dead_fractions[0] > 0.3:
            return "reinitialize"

    # 3. reduce_learning_rate
        strictly_increasing = all(
            dead_fractions[i] > dead_fractions[i - 1]
            for i in range(1, len(dead_fractions))
        )

        if strictly_increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"

    # 4. healthy (strong condition)
        if max(dead_fractions) < 0.1:
            return "healthy"

    # 5. default
        return "healthy"