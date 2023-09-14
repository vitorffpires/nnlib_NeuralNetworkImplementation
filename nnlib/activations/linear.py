import numpy as np
from activation import Activation
class Linear(Activation):
    """
    Função de ativação Linear.
    """

    def forward(self, input_data):
        """
        Retorna os dados de entrada fornecidos (função identidade).

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Os próprios dados de entrada.
        """
        return input_data

    def backward(self, d_output):
        """
        Calcula a derivada da função Linear em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função Linear.
        """
        return np.ones_like(d_output)