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
    def activate(self, input_data):
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
    def derivate(self, d_output):
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
