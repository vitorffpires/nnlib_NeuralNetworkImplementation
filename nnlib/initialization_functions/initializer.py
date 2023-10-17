from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    """
    Abstract base class for weight initialization methods in neural networks.
    """
    
    @abstractmethod
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Abstract method to initialize weights for a layer.
        
        Parameters:
        - input_dim (int): Input dimension for the layer.
        - n_units (int): Number of units in the layer.
        
        Returns:
        np.array:
            Initialized weight matrix.
        """
        pass