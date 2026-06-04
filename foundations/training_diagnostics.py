import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats=[]
        with torch.no_grad():
            for layer in model.children():
                x=layer(x)
                if isinstance(layer,nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)
                    dead_fraction=round((x <= 0).all(dim=0).float().mean().item(), 4)
                    stats.append({
                    "mean": mean,
                    "std": std,
                    "dead_fraction": dead_fraction,
                })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        stats=[]
        out=x
        ll=[]
        for layer in model.children():
            out=layer(out)
            if isinstance(layer,nn.Linear):
                ll.append(layer)
        loss_fn=nn.MSELoss()
        loss=loss_fn(out,y)
        loss.backward()
        for layer in ll:
            g=layer.weight.grad
            stats.append({
            "mean": round(g.mean().item(), 4),
            "std": round(g.std().item(), 4),
            "norm": round(torch.norm(g).item(), 4),
        })
        return stats
    def diagnose(self, activation_stats, gradient_stats):
        if any(layer["dead_fraction"] > 0.5 for layer in activation_stats):
            return "dead_neurons"
        if any(layer["norm"] > 1000 for layer in gradient_stats):
            return "exploding_gradients"
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"
        if any(layer["std"] < 0.1 for layer in activation_stats):
            return "vanishing_gradients"
        if any(layer["std"] > 10.0 for layer in activation_stats):
            return "exploding_gradients"
        return "healthy"
