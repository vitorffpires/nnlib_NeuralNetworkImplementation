from nnlib.layers.layer import Layer
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    Attributes:
    - learning_rate (float): The learning rate for the optimizer.
    """
    def __init__(self, learning_rate: float = 0.01, dropout: float = .0) -> None:
        """
        Initialize the Optimizer with a specified learning rate.
        
        Parameters:
        - learning_rate (float, optional): The learning rate for the optimizer. Default is 0.01.
        """
        self.learning_rate = learning_rate
        self.dropout = dropout
    
        
    @abstractmethod
    def update(self, layer: Layer) -> None:
        """
        Abstract method to update the weights of a given layer.
        
        Parameters:
        - layer (Layer): The layer whose weights need to be updated.
        """
        pass