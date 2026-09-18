class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        der=2*init
        optim=init
        for _ in range(iterations):   
            optim-=learning_rate*der
            der=2*optim 
        return round(optim,5) 
