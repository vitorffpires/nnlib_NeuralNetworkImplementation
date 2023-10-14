from nnlib.activation_functions.activation import Activation
import numpy as np

class Linear(Activation):
    """
    Linear Activation Function.

    The linear activation function, also known as the identity function, simply returns the input value as is.
    It does not introduce any non-linearity to the network.

    Methods:
        activate(x: np.array) -> np.array:
            Compute the output of the linear activation function for the given input.

        derivative(x: np.array) -> np.array:
            Compute the derivative of the linear activation function with respect to its input.
            This function always returns a constant value of 1.

    Args:
        None
    """

    def activate(self, x: np.array) -> np.array:
        """
        Compute the output of the linear activation function for the given input.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Output equal to the input value x.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return x

    def derivate(self, x: np.array) -> np.array:
        """
        Compute the derivative of the linear activation function with respect to its input.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Derivative of the linear activation function always equal to 1.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return np.ones_like(x)
