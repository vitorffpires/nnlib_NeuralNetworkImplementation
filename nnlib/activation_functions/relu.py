from nnlib.activation_functions.activation import Activation
import numpy as np

class ReLu(Activation):
    """
    Rectified Linear Unit (ReLU) Activation Function.

    ReLU is a piecewise linear function that outputs the input directly if it's positive, 
    otherwise, it outputs zero. It has become the default activation function for many 
    types of neural networks because a model that uses it is easier to train and often 
    achieves better performance.

    Inherits:
        Activation: Abstract base class for activation functions.
    """

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the ReLU activation function for the given input.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Output after applying the ReLU activation function. Values less than 0 are set to 0.
        """
        return np.maximum(0, x)
    
    def derivate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the ReLU activation function with respect to its input.

        This is typically used during the backpropagation step in training neural networks.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Derivative of the ReLU activation function with respect to its input. 
                        Returns 1 for values greater than 0, and 0 otherwise.
        """
        return np.where(x > 0, 1, 0)
