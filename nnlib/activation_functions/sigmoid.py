from nnlib.activation_functions.activation import Activation
import numpy as np

class Sigmoid(Activation):
    """
    Sigmoid activation function for neural networks.
    
    The sigmoid function is defined as:
        f(x) = 1 / (1 + exp(-x))
    Its derivative is:
        f'(x) = f(x) * (1 - f(x))
    """

    def activate(self, x: np.array) -> np.array:
        """
        Compute the sigmoid activation for the given input.
        
        Parameters:
        - x (np.array): Input data. Must be a 2D array.
        
        Returns:
        np.array:
            Activated values using the sigmoid function.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return 1 / (1 + np.exp(-x))
    
    def derivate(self, x: np.array) -> np.array:   
        """
        Compute the derivative of the sigmoid function for the given input.
        
        Parameters:
        - x (np.array): Input data. Must be a 2D array.
        
        Returns:
        np.array:
            Derivative values of the sigmoid function.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        function = self.activate(x = x)
        return function * (1 - function)
