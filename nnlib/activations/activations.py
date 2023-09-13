import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    """
    Esta é uma classe base para funções de ativação. Cada função de ativação deve implementar
    os métodos para calcular a ativação e sua derivada.
    """

    def __init__(self):
        """
        Construtor da classe Activation.
        """
        pass

    @abstractmethod
    def forward(self, input_data):
        """
        Calcula a ativação para os dados de entrada fornecidos.

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Valores de ativação calculados.
        """
        raise NotImplementedError("O método forward precisa ser implementado.")

    @abstractmethod
    def backward(self, d_output):
        """
        Calcula a derivada da ativação em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função de ativação.
        """
        raise NotImplementedError("O método backward precisa ser implementado.")

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


class Softmax(Activation):
    """
    Função de ativação Softmax.
    """

    def forward(self, input_data):
        """
        Calcula a ativação Softmax para os dados de entrada fornecidos.

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Valores de ativação calculados usando Softmax.
        """
        exp_data = np.exp(input_data - np.max(input_data))
        return exp_data / np.sum(exp_data, axis=-1, keepdims=True)

    def backward(self, d_output):
        """
        Calcula a derivada da função Softmax em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função Softmax.
        """
        s = self.forward(d_output)
        return s * (1 - s)


class LeakyReLU(Activation):
    """
    Função de ativação Leaky ReLU.
    """
    def __init__(self, alpha=0.01):
        """
        Construtor da função LeakyReLU.

        Parameters:
        - alpha: float
            Coeficiente que determina o valor "leaky".
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input_data):
        """
        Calcula a ativação LeakyReLU para os dados de entrada fornecidos.

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Valores de ativação calculados usando LeakyReLU.
        """
        return np.where(input_data > 0, input_data, self.alpha * input_data)

    def backward(self, d_output):
        """
        Calcula a derivada da função LeakyReLU em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função LeakyReLU.
        """
        return np.where(d_output > 0, 1, self.alpha)


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
