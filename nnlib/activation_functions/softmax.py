from nnlib.activation_functions.activation import Activation
import numpy as np

class Softmax(Activation):

    def activate(self, x: np.array) -> np.array:
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        x = x - np.max(x, axis=1, keepdims=True)  # To prevent overflow
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def derivate(self, x: np.array) -> np.array:
        if x.ndim != 2:
            raise ValueError(f"on activation, x must be a 2D array, got {x.ndim}")
        return np.ones_like(x)