import numpy as np
from nnlib.layers.layer import Layer
from nnlib.optimization_functions.optimizer import Optimizer


class AdaptiveMomentEstimation(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimization algorithm.
    
    Attributes:
    - beta_1 (float): Exponential decay rate for the first moment estimates.
    - beta_2 (float): Exponential decay rate for the second moment estimates.
    - epsilon (float): Small constant to prevent division by zero.
    - t (int): Time step counter.
    """    
     
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float =0.999, epsilon: float = 1e-15) -> None:
        """
        Initialize the Adam optimizer with specified parameters.
        
        Parameters:
        - learning_rate (float, optional): The learning rate for the optimizer. Default is 0.01.
        - beta_1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
        - beta_2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
        - epsilon (float, optional): Small constant to prevent division by zero. Default is 1e-15.
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0     # Time step counter

    def update(self, layer: Layer) -> None:
        """
        Update the weights and biases of a given layer using the Adam optimization algorithm.
        
        Parameters:
        - layer (Layer): The layer whose weights and biases need to be updated.
        """
        if (layer.m_w is None) or (layer.v_w is None) or (layer.m_b is None) or (layer.v_b is None):
            layer.m_w = np.zeros_like(layer.get_weights()['weights'])
            layer.v_w = np.zeros_like(layer.get_weights()['weights'])
            layer.m_b = np.zeros_like(layer.get_weights()['bias'])
            layer.v_b = np.zeros_like(layer.get_weights()['bias'])
        
        # Update time step
        self.t += 1
        
        if layer.m_w.shape != layer.derivative_weights.shape:
            raise ValueError(f"Shape mismatch: m has shape {layer.m_w.shape} but derivative_weights has shape {layer.derivative_weights.shape}")
        if layer.v_w.shape != layer.derivative_weights.shape:
            raise ValueError(f"Shape mismatch: v has shape {layer.m_w.shape} but derivative_weights has shape {layer.derivative_weights.shape}")
        if layer.m_b.shape != layer.derivative_bias.shape:
            raise ValueError(f"Shape mismatch: m_b has shape {layer.m_b.shape} but derivative_bias has shape {layer.derivative_bias.shape}")
        if layer.v_b.shape != layer.derivative_bias.shape:
            raise ValueError(f"Shape mismatch: v_b has shape {layer.m_b.shape} but derivative_bias has shape {layer.derivative_bias.shape}")

        # Update moving averages of gradients and squared gradients
        layer.m_w = self.beta_1 * layer.m_w + (1. - self.beta_1) * layer.derivative_weights
        layer.v_w = self.beta_2 * layer.v_w + (1. - self.beta_2) * np.square(layer.derivative_weights)
        layer.m_b = self.beta_1 * layer.m_b + (1. - self.beta_1) * layer.derivative_bias
        layer.v_b = self.beta_2 * layer.v_b + (1. - self.beta_2) * np.square(layer.derivative_bias)
        
        # Compute corrected moving averages
        m_corrected_w = layer.m_w / (1. - self.beta_1**self.t)
        v_corrected_w = layer.v_w / (1. - self.beta_2**self.t)
        m_corrected_b = layer.m_b / (1. - self.beta_1**self.t)
        v_corrected_b = layer.v_b / (1. - self.beta_2**self.t)
        
        # Update weights and biases
        layer.weights = layer.weights - (self.learning_rate * m_corrected_w / (np.sqrt(v_corrected_w) + self.epsilon))
        layer.bias = layer.bias - (self.learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + self.epsilon))
