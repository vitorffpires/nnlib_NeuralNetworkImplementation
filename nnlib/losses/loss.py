import numpy as np
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


class MSE(Loss):
    """
    Função de perda de erro quadrático médio (MSE).
    """

    def forward(self, y_pred, y_true):
        """
        Calcula o MSE entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        """
        Calcula o gradiente do MSE em relação às previsões y_pred.
        """
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(Loss):
    """
    Função de perda de entropia cruzada binária.
    """

    def forward(self, y_pred, y_true):
        """
        Calcula a entropia cruzada entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        epsilon = 1e-15  # para evitar o log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred, y_true):
        """
        Calcula o gradiente da entropia cruzada em relação às previsões y_pred.
        """
        epsilon = 1e-15  # para evitar divisão por zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
