import numpy as np
from nnlib.layers.layer import Layer
from nnlib.optimization_functions.optimizer import Optimizer

class StochastciGradientDescent(Optimizer):
    
    def update(self, layer: Layer) -> None:
        """
        Update the weights of the given layer using SGD.
        
        Parameters:
        - layer: Layer
            The layer whose parameters are to be updated.
        """
        # Update weights using the gradients and learning rate
        layer.weights = layer.weights - (self.learning_rate * layer.derivative_weights)

        layer.biases = layer.biases - (self.learning_rate * layer.derivative_bias)
