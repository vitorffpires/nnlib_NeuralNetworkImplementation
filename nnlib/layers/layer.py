from abc import ABC, abstractmethod
import numpy as np
from nnlib.activations import ActivationFunction  # Supondo que a implementação anterior tenha esta estrutura

class Layer(ABC):
    """
    Classe base para todas as camadas.
    """

    def __init__(self):
        pass

    @abstractmethod
    def forward_pass(self, input_data):
        pass

    @abstractmethod
    def backward_pass(self, output_grad):
        pass

class Dense(Layer):
    """
    Camada Densa (ou Totalmente Conectada).
    """

    def __init__(self, input_dim, output_dim, activation_function):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.activation = ActivationFunction(activation_function)

    def forward_pass(self, input_data):
        self.input_data = input_data
        self.output = self.activation.activate(np.dot(input_data, self.weights) + self.bias)
        return self.output

    def backward_pass(self, output_grad):
        d_activation = self.activation.derivate(self.output) * output_grad
        d_weights = np.dot(self.input_data.T, d_activation)
        d_bias = np.sum(d_activation, axis=0, keepdims=True)
        d_input_data = np.dot(d_activation, self.weights.T)
        
        # Atualização dos pesos e bias será realizada pelo otimizador
        self.d_weights = d_weights
        self.d_bias = d_bias
        
        return d_input_data

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
