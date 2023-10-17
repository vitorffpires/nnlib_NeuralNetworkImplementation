import numpy as np
from abc import ABC, abstractmethod
from nnlib.activation_functions.activation import Activation
from nnlib.activation_functions.linear import Linear
from nnlib.initialization_functions.uniform import Uniform


class Layer(ABC):
    """
    Abstract base class for layers in a neural network.
    
    Attributes:
    - n_units (int): Number of units in the layer.
    - activation (Activation): Activation function for the layer.
    - input_dim (int, optional): Input dimension for the layer.
    - m (np.array, optional): First moment estimate for Adam optimizer.
    - v (np.array, optional): Second moment estimate for Adam optimizer.
    - is_initialized (bool): Flag to check if the layer is initialized.
    """

    def __init__(self, 
                 n_units: int,
                 input_dim: int = None,
                 activation: Activation = Linear()
                ) -> None:
        """
        Initialize the layer with given parameters.
        
        Parameters:
        - n_units (int): Number of units in the layer.
        - input_dim (int, optional): Input dimension for the layer. Default is None.
        - activation (Activation, optional): Activation function for the layer. Default is Linear.
        """
        self.n_units = n_units
        self.activation = activation
        self.input_dim = input_dim
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.is_initialized = False

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        """
        Abstract method for forward pass.
        
        Parameters:
        - x (np.array): Input data.
        
        Returns:
        np.array:
            Output after forward pass.
        """
        pass

    @abstractmethod
    def backward(self, loss_derivative: np.array) -> np.array:
        """
        Abstract method for backward pass.
        
        Parameters:
        - loss_derivative (np.array): Derivative of the loss.
        
        Returns:
        np.array:
            Gradient of the loss with respect to the input.
        """
        pass

    @abstractmethod
    def get_weights(self) -> dict:
        """
        Abstract method to get the weights of the layer.
        
        Returns:
        dict:
            Weights of the layer.
        """
        pass


    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """
        Abstract method to set the weights of the layer.
        
        Parameters:
        - weights (dict): Weights to be set.
        """
        pass