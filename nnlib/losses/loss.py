from abc import ABC, abstractmethod

class Loss(ABC):
    """
    Classe base para todas as funções de perda.     
    """

    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        Calcula a perda entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, y_pred, y_true):
        """
        Calcula o gradiente da perda em relação às previsões y_pred.
        """
        raise NotImplementedError
