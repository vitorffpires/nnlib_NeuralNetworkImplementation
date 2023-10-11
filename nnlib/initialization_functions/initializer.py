from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    
    @abstractmethod
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        pass