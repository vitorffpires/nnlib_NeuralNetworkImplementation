from metrics import Metric
import numpy as np
class Precision(Metric):
    """
    Métrica de Precisão.
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula a precisão entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / (predicted_positives + 1e-10)  # adicionado 1e-10 para evitar divisão por zero