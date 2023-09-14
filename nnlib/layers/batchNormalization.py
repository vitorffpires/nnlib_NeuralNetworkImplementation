import numpy as np
from Layer import Layer
class BatchNormalization(Layer):
    """
    Camada de Normalização em Lote.
    """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.running_mean = 0
        self.running_var = 0

    def forward_pass(self, input_data, training=True):
        if training:
            batch_mean = np.mean(input_data, axis=0)
            batch_var = np.var(input_data, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.normalized_data = (input_data - batch_mean) / np.sqrt(batch_var + 1e-8)
            return self.normalized_data
        else:
            return (input_data - self.running_mean) / np.sqrt(self.running_var + 1e-8)

    def backward_pass(self, output_grad):
        # Esta é uma simplificação, a implementação completa é mais complexa
        return output_grad
