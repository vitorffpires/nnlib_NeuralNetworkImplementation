import numpy as np
from abc import ABC, abstractmethod
class Optimizer(ABC):
    """
    Classe base para todos os otimizadores.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, gradient):
        """
        Atualiza os pesos com base no gradiente.
        """
        raise NotImplementedError