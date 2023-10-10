from nnlib.activation_functions.activation import Activation
import numpy as np

class Sigmoid(Activation):
    """
    Sigmoid Activation Function.

    The sigmoid activation function, also known as the logistic function, maps input values to the range (0, 1).
    It is commonly used in the hidden layers of neural networks for introducing non-linearity.

    Methods:
        activate(x: np.ndarray) -> np.ndarray:
            Compute the output of the sigmoid activation function for the given input.

        derivative(x: np.ndarray) -> np.ndarray:
            Compute the derivative of the sigmoid activation function with respect to its input.

    Args:
        None
    """

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the output of the sigmoid activation function for the given input.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Output values in the range (0, 1) after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
    
    def derivate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid activation function with respect to its input.

        Args:
            x (np.ndarray): Input data, typically the output from a previous layer in a neural network.

        Returns:
            np.ndarray: Derivative of the sigmoid activation function at the given input values.
                        It is calculated as sigmoid(x) * (1 - sigmoid(x)).
        """
        sigmoid = self.activate(x)
        return sigmoid * (1 - sigmoid)
