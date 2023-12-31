from nnlib.activation_functions.activation import Activation
import numpy as np

class LeakyReLu(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLu) Activation Function.

    The Leaky ReLu activation function is a variant of the Rectified Linear Unit (ReLu) function,
    allowing a small, non-zero gradient for negative inputs to prevent neurons from dying during training.

    Args:
        alpha (float, optional): The slope of the leaky part for negative inputs.
            A small positive value (e.g., 0.01) is typically used. Defaults to 0.01.

    Methods:
        activate(x: np.array) -> np.array:
            Compute the output of the Leaky ReLu activation function for the given input.

        derivate(x: np.array) -> np.array:
            Compute the derivative of the Leaky ReLu activation function with respect to its input.
            This is typically used during the backpropagation step in training neural networks.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initialize a LeakyReLu instance.

        Args:
            alpha (float, optional): The slope of the leaky part for negative inputs.
                A small positive value (e.g., 0.01) is typically used. Defaults to 0.01.
        """
        self.alpha = alpha

    def activate(self, x: np.array) -> np.array:
        """
        Compute the output of the Leaky ReLu activation function for the given input.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Output after applying the Leaky ReLu activation function.
                        Values greater than or equal to zero remain unchanged, while
                        negative values are scaled by the specified alpha.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return np.where(x > 0, x, self.alpha * x)

    def derivate(self, x: np.array) -> np.array:
        """
        Compute the derivative of the Leaky ReLu activation function with respect to its input.

        This is typically used during the backpropagation step in training neural networks.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Derivative of the Leaky ReLu activation function with respect to its input. 
                        Returns 1 for values greater than zero and the specified alpha for
                        values less than or equal to zero.
        """
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return np.where(x > 0, 1, self.alpha)
