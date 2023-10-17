from nnlib.activation_functions.activation import Activation
import numpy as np

class Softmax(Activation):
    """
    Implements the Softmax activation function.
    
    The Softmax function is commonly used in the output layer of a neural network for multi-class 
    classification problems. It converts the raw output scores (logits) into probabilities, such that 
    the sum of the probabilities for each row equals 1.
    
    Methods:
    - activate(x): Applies the Softmax function to the input array.
    - derivate(x): Computes the derivative of the Softmax function. For the Softmax function, 
                   the derivative is simply an array of ones with the same shape as the input.
    """

    def activate(self, x: np.array) -> np.array:
        """
        Apply the Softmax function to the input array.
        
        Parameters:
        - x: np.array
            Input array (logits) to which the Softmax function should be applied.
        
        Returns:
        np.array:
            Array of probabilities after applying the Softmax function.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        x = x - np.max(x, axis=1, keepdims=True)  # To prevent overflow
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def derivate(self, x: np.array) -> np.array:
        """
        Compute the derivative of the Softmax function.
        
        For the Softmax function, the derivative is simply an array of ones with the same shape as the input.
        
        Parameters:
        - x: np.array
            Input array (logits).
        
        Returns:
        np.array:
            Derivative of the Softmax function.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return np.ones_like(x)