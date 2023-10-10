from nnlib.initialization_functions.initializer import Initializer
import numpy as np

class He(Initializer):

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        std = np.sqrt(2. / input_dim)
        weight_matrix = np.random.normal(0, std, size=(input_dim, n_units))
        return weight_matrix
    
    def initialize_bias(self, n_units):
        return np.zeros((1, n_units))