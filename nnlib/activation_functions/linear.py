from nnlib.activation_functions.activation import Activation
import numpy as np

class Linear(Activation):
    """
    Linear Activation Function.

    The linear activation function, also known as the identity function, simply returns the input value as is.
    It does not introduce any non-linearity to the network.

    Methods:
        activate(x: np.ndarray) -> np.ndarray:
            Compute the output of the linear activation function for the given input.

        derivative(x: np.ndarray) -> np.ndarray:
            Compute the derivative of the linear activation function with respect to its input.
            This function always returns a constant value of 1.

    Args:
        None
    """

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the linear activation function for the given input.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Output equal to the input value x.
        """
        return x

    def derivate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the linear activation function with respect to its input.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Derivative of the linear activation function always equal to 1.
        """
        return np.ones_like(x)
