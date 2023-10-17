from nnlib.activation_functions.activation import Activation
import numpy as np

class Sigmoid(Activation):

    def activate(self, x: np.array) -> np.array:
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return 1 / (1 + np.exp(-x))
    
    def derivate(self, x: np.array) -> np.array:   
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        function = self.activate(x = x)
        return function * (1 - function)
