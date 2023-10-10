import numpy as np
from abc import ABC, abstractmethod
from nnlib.activation_functions.linear import Linear
from nnlib.initialization_functions.uniform import Uniform


class Layer(ABC):

    def __init__(self, 
                 n_units: int, 
                 activation: object = Linear(), 
                 initializer: object = Uniform(),
                 has_bias: bool = True,
                 input_dim: int = None,
                ) -> None:
        self.n_units = n_units
        self.activation = activation
        self.initializer = initializer
        self.has_bias = has_bias
        self.input_dim = input_dim
        self.is_initialized = False
        
        
    def initialize(self) -> None:
        if not self.is_initialized:
            self.weights = self.initializer.initialize_weights(self.input_dim, self.n_units)
            self.biases = self.initializer.initialize_bias(self.n_units)
            self.is_initialized = True

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass


    @abstractmethod
    def backward(self, loss_derivative: np.array) -> np.array:
        pass


    @abstractmethod
    def get_weights(self) -> dict:
        pass


    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        pass