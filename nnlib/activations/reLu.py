import numpy as np
from activation import Activation
class ReLU(Activation):
    """
    Função de ativação ReLU (Rectified Linear Unit).
    """

    def forward(self, input_data):
        """
        Calcula a ativação ReLU para os dados de entrada fornecidos.
        """
        return np.maximum(0, input_data)

    def backward(self, d_output):
        """
        Calcula a derivada da função ReLU em relação à saída.
        """
        return np.where(d_output > 0, 1, 0)