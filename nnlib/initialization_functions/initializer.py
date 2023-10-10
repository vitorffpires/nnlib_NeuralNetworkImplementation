from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    
    @abstractmethod
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        pass

    @abstractmethod
    def initialize_bias(self, n_units: int) -> np.array:
        return np.zeros((1, n_units))