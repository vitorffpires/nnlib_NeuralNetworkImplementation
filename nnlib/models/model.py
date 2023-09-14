from abc import ABC, abstractmethod
class Model(ABC):
    """
    Classe base para todos os modelos.
    """

    def __init__(self):
        # Lista para armazenar as camadas
        self.layers = []

    def add(self, layer):
        """
        Adiciona uma camada ao modelo.
        """
        self.layers.append(layer)

    @abstractmethod
    def compile(self, optimizer, loss_function, metrics=[]):
        """
        Compila o modelo com um otimizador, função de perda e métricas.
        """
        pass

    @abstractmethod
    def fit(self, X, y, epochs, batch_size):
        """
        Treina o modelo usando os dados fornecidos.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Avalia o modelo nos dados fornecidos.
        """
        pass

    @abstractmethod
    def save(self, filename):
        """
        Salva o modelo no caminho especificado.
        """
        pass

    @abstractmethod
    def load(self, filename):
        """
        Carrega um modelo do caminho especificado.
        """
        pass

    @abstractmethod
    def log(self, message):
        """
        Faz log de mensagens ou métricas importantes.
        """
        pass