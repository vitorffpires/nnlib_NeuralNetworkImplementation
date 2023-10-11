import numpy as np
from abc import ABC, abstractmethod
from nnlib.activation_functions.linear import Linear
from nnlib.initialization_functions.uniform import Uniform


class Layer(ABC):

    def __init__(self, 
                 n_units: int, 
                 input_dim: int = None,
                 activation: object = Linear()
                ) -> None:
        self.n_units = n_units
        self.activation = activation
        self.input_dim = input_dim
        self.is_initialized = False

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