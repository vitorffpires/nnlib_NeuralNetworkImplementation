import numpy as np
from nnlib.layers.layer import Layer
from nnlib.optimization_functions.optimizer import Optimizer
class AdaptiveMomentEstimation(Optimizer):
    
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.8, beta_2: float = 0.8, epsilon: float = 1e-15) -> None:
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0     # Time step counter

    def update(self, layer: Layer) -> None:
        """
        Update the weights of the given layer using Adam.
        
        Parameters:
        - layer: Layer
            The layer whose parameters are to be updated.
        """
        # Initialize moving averages if first update
        if (layer.m is None) or (layer.v is None):
            layer.m = np.zeros_like(layer.get_weights())
            layer.v = np.zeros_like(layer.get_weights())
        
        # Update time step
        self.t += 1
        
        if layer.m.shape != layer.derivative_weights.shape:
            raise ValueError(f"Shape mismatch: m has shape {layer.m.shape} but derivative_weights has shape {layer.derivative_weights.shape}")
        if layer.v.shape != layer.derivative_weights.shape:
            raise ValueError(f"Shape mismatch: v has shape {layer.m.shape} but derivative_weights has shape {layer.derivative_weights.shape}")

        # Update moving averages of gradients and squared gradients
        layer.m = self.beta_1 * layer.m + (1. - self.beta_1) * layer.derivative_weights
        layer.v = self.beta_2 * layer.v + (1. - self.beta_2) * np.square(layer.derivative_weights)
        
        # Compute corrected moving averages
        m_corrected = layer.m / (1. - self.beta_1**self.t)
        v_corrected = layer.v / (1. - self.beta_2**self.t)
        
        # Update weights
        layer.weights = layer.weights - (self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon))    
