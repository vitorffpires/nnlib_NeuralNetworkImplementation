import numpy as np
from nnlib.layers.layer import Layer
from nnlib.optimization_functions.optimizer import Optimizer

class StochastciGradientDescent(Optimizer):
    
    def update(self, layer: Layer) -> None:
        """
        Update the weights and biases of the given layer using SGD.
        
        Parameters:
        - layer: Layer
            The layer whose parameters are to be updated.
        """
        # Update weights and biases using the gradients and learning rate
        layer.weights -= self.learning_rate * layer.derivative_weights
        if layer.has_bias:
            layer.biases -= self.learning_rate * layer.derivative_biases
