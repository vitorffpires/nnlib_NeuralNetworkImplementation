from abc import ABC, abstractmethod
class Regularization(ABC):
    """
    Classe base para regularizações. Esta não é um otimizador, mas pode ser usado junto com um.
    """

    def __init__(self, lambda_value):
        self.lambda_value = lambda_value

    @abstractmethod
    def regularization_term(self, weights):
        """
        Calcula o termo de regularização.
        """
        raise NotImplementedError