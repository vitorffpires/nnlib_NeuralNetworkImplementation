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
        batch_size, n_features = x.shape
        softmax = self.compute(x)
        
        # Initialize a 3D array to store the Jacobian matrices for each sample
        jacobian_batch = np.zeros((batch_size, n_features, n_features))
        
        # Compute the Jacobian matrix for each sample in the batch
        for i in range(batch_size):
            s = softmax[i].reshape(-1, 1)
            jacobian_batch[i] = np.diagflat(s) - np.dot(s, s.T)
        
        return jacobian_batch 