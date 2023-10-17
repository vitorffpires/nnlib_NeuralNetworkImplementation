from nnlib.initialization_functions.initializer import Initializer
import numpy as np

class Uniform(Initializer):
    """
    Uniform initialization method for neural networks.
    
    Attributes:
    - low_limit (float): Lower limit for the uniform distribution.
    - high_limit (float): Upper limit for the uniform distribution.
    """

    def __init__(self, low_limit: float = -1.0, high_limit: float = 1.0) -> None:
        """
        Initialize the Uniform initializer with given parameters.
        
        Parameters:
        - low_limit (float, optional): Lower limit for the uniform distribution. Default is -1.0.
        - high_limit (float, optional): Upper limit for the uniform distribution. Default is 1.0.
        """
        self.low_limit = low_limit
        self.high_limit = high_limit

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initialize weights using a uniform distribution.
        
        Parameters:
        - input_dim (int): Input dimension for the layer.
        - n_units (int): Number of units in the layer.
        
        Returns:
        np.array:
            Initialized weight matrix.
        """
        weight_matrix = np.random.uniform(self.low_limit, self.high_limit, (input_dim, n_units))
        return weight_matrix