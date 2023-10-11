from nnlib.initialization_functions.initializer import Initializer
import numpy as np

class Normal(Initializer):

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        weight_matrix = np.random.normal(self.mean, self.std, (input_dim, n_units))
        return weight_matrix