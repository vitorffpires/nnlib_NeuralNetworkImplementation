from nnlib.activation_functions.activation import Activation
import numpy as np

class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) Activation Function.

    The Tanh activation function maps input values to the range (-1, 1) and is often used in neural networks
    to introduce non-linearity.

    Methods:
        activate(x: np.ndarray) -> np.ndarray:
            Compute the output of the Tanh activation function for the given input.

        derivative(x: np.ndarray) -> np.ndarray:
            Compute the derivative of the Tanh activation function with respect to its input.

    Args:
        None
    """

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the Tanh activation function for the given input.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Output values in the range (-1, 1) after applying the Tanh function.
        """
        return np.tanh(x)

    def derivate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the Tanh activation function with respect to its input.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Derivative of the Tanh activation function at the given input values.
                        It is calculated as 1 - tanh(x)^2.
        """
        return 1 - np.tanh(x)**2
