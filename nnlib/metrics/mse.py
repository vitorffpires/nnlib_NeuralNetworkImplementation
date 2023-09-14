from metrics import Metric
import numpy as np
class MSE(Metric):
    """
    Métrica de Erro Quadrático Médio.
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula o MSE entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        return np.mean((y_pred - y_true) ** 2)