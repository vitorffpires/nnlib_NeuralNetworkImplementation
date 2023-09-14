from loss import Loss
import numpy as np
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