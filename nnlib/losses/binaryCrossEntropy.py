from loss import Loss
import numpy as np
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