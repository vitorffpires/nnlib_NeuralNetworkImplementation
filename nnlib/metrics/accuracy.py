from metrics import Metric
import numpy as np
class Accuracy(Metric):
    """
    Métrica de Acurácia.
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula a acurácia entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        return np.mean(y_pred == y_true)