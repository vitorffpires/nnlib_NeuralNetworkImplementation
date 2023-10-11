from nnlib.activation_functions.activation import Activation
import numpy as np

class Softmax(Activation):

    def activate(self, input: np.array) -> np.array:
        input = input - np.max(input, axis=1, keepdims=True)  # To prevent overflow
        exp = np.exp(input)
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        return softmax

    def derivate(self, input: np.array) -> np.array:
        batch_size, n_features = input.shape
        softmax = self.compute(input)
        
        # Initialize a 3D array to store the Jacobian matrices for each sample
        jacobian_batch = np.zeros((batch_size, n_features, n_features))
        
        # Compute the Jacobian matrix for each sample in the batch
        for i in range(batch_size):
            s = softmax[i].reshape(-1, 1)
            jacobian_batch[i] = np.diagflat(s) - np.dot(s, s.T)
        
        return jacobian_batch 