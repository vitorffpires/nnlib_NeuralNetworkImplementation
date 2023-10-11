import numpy as np
from nnlib.layers.layer import Layer
from nnlib.optimization_functions.optimizer import Optimizer
class AdaptiveMomentEstimation(Optimizer):
    
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None  # Moving average of the gradients
        self.v = None  # Moving average of the square of the gradients
        self.t = 0     # Time step counter
    
    def update(self, layer: Layer) -> None:
        """
        Update the weights and biases of the given layer using Adam.
        
        Parameters:
        - layer: Layer
            The layer whose parameters are to be updated.
        """
        # Initialize moving averages if first update
        if self.m is None:
            self.m = {'weights': np.zeros_like(layer.weights)}
            self.v = {'weights': np.zeros_like(layer.weights)}
        
        # Update time step
        self.t += 1
        
        # Update moving averages of gradients and squared gradients
        self.m['weights'] = self.beta_1 * self.m['weights'] + (1. - self.beta_1) * layer.derivative_weights
        self.v['weights'] = self.beta_2 * self.v['weights'] + (1. - self.beta_2) * np.square(layer.derivative_weights)
        
        # Compute bias-corrected moving averages
        m_hat = {param: m_val / (1. - self.beta_1**self.t) for param, m_val in self.m.items() if m_val is not None}
        v_hat = {param: v_val / (1. - self.beta_2**self.t) for param, v_val in self.v.items() if v_val is not None}
        
        # Update weights and biases
        layer.weights = -1 * self.learning_rate * m_hat['weights'] / (np.sqrt(v_hat['weights']) + self.epsilon)
