import numpy as np
from layer import Layer
class Dropout(Layer):
    """
    Camada de Dropout.
    """

    def __init__(self, rate):
        self.rate = rate

    def forward_pass(self, input_data, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            return input_data * self.mask
        return input_data

    def backward_pass(self, output_grad):
        return output_grad * self.mask