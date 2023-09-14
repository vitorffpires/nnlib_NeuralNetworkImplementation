import numpy as np
from activation import Activation
class Sigmoid(Activation):
    """
    Função de ativação Sigmoid.
    """

    def forward(self, input_data):
        """
        Calcula a ativação Sigmoid para os dados de entrada fornecidos.
        """
        return 1 / (1 + np.exp(-input_data))

    def backward(self, d_output):
        """
        Calcula a derivada da função Sigmoid em relação à saída.
        """
        sigmoid = self.forward(d_output)
        return sigmoid * (1 - sigmoid)