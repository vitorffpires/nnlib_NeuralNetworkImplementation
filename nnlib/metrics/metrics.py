import numpy as np

class Metric:
    """
    Classe base para todas as métricas.
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula a métrica entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        raise NotImplementedError


class Accuracy(Metric):
    """
    Métrica de Acurácia.
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula a acurácia entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        return np.mean(y_pred == y_true)


class MSE(Metric):
    """
    Métrica de Erro Quadrático Médio.
    """

    def calculate(self, y_pred, y_true):
        """
        Calcula o MSE entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        return np.mean((y_pred - y_true) ** 2)


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
