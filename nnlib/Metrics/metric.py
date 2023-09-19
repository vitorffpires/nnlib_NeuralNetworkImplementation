import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Classe base para todas as métricas.
    """
    @abstractmethod
    def calculate(self, y_pred, y_true):
        """
        Calcula a métrica entre as previsões y_pred e os verdadeiros rótulos y_true.
        """
        raise NotImplementedError