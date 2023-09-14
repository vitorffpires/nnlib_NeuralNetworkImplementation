from metrics import Metric
import numpy as np
class R2(Metric):
    """
    Métrica R^2 (coeficiente de determinação).
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula o R^2 entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))  # adicionado 1e-10 para evitar divisão por zero
