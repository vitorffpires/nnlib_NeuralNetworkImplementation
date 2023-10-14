from nnlib.activation_functions.activation import Activation
import numpy as np

class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) Activation Function.

    The Tanh activation function maps input values to the range (-1, 1) and is often used in neural networks
    to introduce non-linearity.

    Methods:
        activate(x: np.array) -> np.array:
            Compute the output of the Tanh activation function for the given input.

        derivative(x: np.array) -> np.array:
            Compute the derivative of the Tanh activation function with respect to its input.

    Args:
        None
    """

    def activate(self, x: np.array) -> np.array:
        """
        Compute the output of the Tanh activation function for the given input.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Output values in the range (-1, 1) after applying the Tanh function.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return (np.exp(2*x)-1) / (np.exp(2*x)+1)

    def derivate(self, x: np.array) -> np.array:
        """
        Compute the derivative of the Tanh activation function with respect to its input.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Derivative of the Tanh activation function at the given input values.
                        It is calculated as 1 - tanh(x)^2.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        x = self.activate(x)
        return 1 - x**2
