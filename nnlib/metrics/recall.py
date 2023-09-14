from metrics import Metric
import numpy as np
class Recall(Metric):
    """
    Métrica de Revocação (Recall).
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula a revocação entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / (actual_positives + 1e-10)  # adicionado 1e-10 para evitar divisão por zero