from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    """
    Abstract base class for activation functions in neural networks.

    This class provides a blueprint for implementing various activation functions.
    Derived classes should implement the `activate` and `derivative` methods.

    Args:
        ABC (class): Inherited from Python's Abstract Base Class (ABC) to ensure that derived 
        classes implement the abstract methods.
    """

    @abstractmethod
    def activate(self, x: np.array) -> np.array:
        """
        Computes the output of the activation function for the given input.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Output after applying the activation function.
        """
        pass

    @abstractmethod
    def derivate(self, x: np.array) -> np.array:
        """
        Computes the derivative of the activation function with respect to its input.

        This is typically used during the backpropagation step in training neural networks.

        Args:
            x (np.array): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.array: Derivative of the activation function with respect to its input.
        """
        pass
