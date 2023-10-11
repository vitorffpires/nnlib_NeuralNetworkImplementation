from nnlib.initialization_functions.initializer import Initializer
import numpy as np

class Uniform(Initializer):

    def __init__(self, low_limit: float = -1.0, high_limit: float = 1.0) -> None:
        self.low_limit = low_limit
        self.high_limit = high_limit

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        weight_matrix = np.random.uniform(self.low_limit, self.high_limit, (input_dim, n_units))
        return weight_matrix