from metrics import Metric
from precision import Precision
from recall import Recall
class F1Score(Metric):
    """
    Métrica F1 Score.
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula o F1 Score entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        precision = Precision().calculate(y_pred, y_true)
        recall = Recall().calculate(y_pred, y_true)
        return 2 * (precision * recall) / (precision + recall + 1e-10)  # adicionado 1e-10 para evitar divisão por zero