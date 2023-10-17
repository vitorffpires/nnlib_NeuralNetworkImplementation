from nnlib.initialization_functions.initializer import Initializer
import numpy as np

class Xavier(Initializer):
    """
    Xavier (Glorot) initialization method for neural networks.
    """

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initialize weights using the Xavier method.
        
        Parameters:
        - input_dim (int): Input dimension for the layer.
        - n_units (int): Number of units in the layer.
        
        Returns:
        np.array:
            Initialized weight matrix.
        """
        std = np.sqrt(1.0 / (input_dim + n_units))
        weight_matrix = np.random.normal(0, std, size=(input_dim, n_units))
        return weight_matrix