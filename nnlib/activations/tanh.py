import numpy as np
from activation import Activation
class Tanh(Activation):
    """
    Função de ativação Tanh (tangente hiperbólica).
    """

    def forward(self, input_data):
        """
        Calcula a ativação Tanh para os dados de entrada fornecidos.

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Valores de ativação calculados usando Tanh.
        """
        return np.tanh(input_data)

    def backward(self, d_output):
        """
        Calcula a derivada da função Tanh em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função Tanh.
        """
        tanh_val = self.forward(d_output)
        return 1.0 - tanh_val**2