import numpy as np
from nnlib.layers.layer import Layer
from nnlib.optimization_functions.optimizer import Optimizer

class StochastciGradientDescent(Optimizer):

    def __init__(self, learning_rate: float = 0.01, dropout: float = 0) -> None:
        super().__init__(learning_rate, dropout)
    
    def update(self, layer: Layer) -> None:
        """
        Update the weights of the given layer using SGD.
        
        Parameters:
        - layer: Layer
            The layer whose parameters are to be updated.
        """
        # Update weights using the gradients and learning rate
        updated_weights = (self.learning_rate * layer.derivative_weights)
        if self.dropout > 0:
            idx_qty = int(len(layer.weights) * self.dropout)
            idx_dropouts = np.random.randint(0, len(layer.weights), idx_qty)
            
            for idx in idx_dropouts:
                updated_weights[idx] = 0

        layer.weights = layer.weights - updated_weights

        layer.biases = layer.biases - (self.learning_rate * layer.derivative_bias)
