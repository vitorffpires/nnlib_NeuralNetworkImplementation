from nnlib.initialization_functions.initializer import Initializer
import numpy as np

class Normal(Initializer):
    """
    Normal (Gaussian) initialization method for neural networks.
    
    Attributes:
    - mean (float): Mean value for the normal distribution.
    - std (float): Standard deviation for the normal distribution.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """
        Initialize the Normal initializer with given parameters.
        
        Parameters:
        - mean (float, optional): Mean value for the normal distribution. Default is 0.0.
        - std (float, optional): Standard deviation for the normal distribution. Default is 1.0.
        """
        self.mean = mean
        self.std = std

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initialize weights using a normal distribution.
        
        Parameters:
        - input_dim (int): Input dimension for the layer.
        - n_units (int): Number of units in the layer.
        
        Returns:
        np.array:
            Initialized weight matrix.
        """
        weight_matrix = np.random.normal(self.mean, self.std, (input_dim, n_units))
        return weight_matrix