from abc import ABC, abstractmethod
import numpy as np

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

class Sequential(Model):
    """
    Modelo sequencial que executa as camadas em sequência.
    """

    def __init__(self):
        super().__init__()

    def compile(self, optimizer, loss_function, metrics=[]):
        # Definir otimizador, função de perda e métricas para o modelo
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def fit(self, X, y, epochs, batch_size):
        # TODO: Implementar lógica de treinamento
        pass

    def evaluate(self, X, y):
        # TODO: Implementar avaliação do modelo
        pass

    def save(self, filename):
        # TODO: Salvar o modelo em um arquivo
        pass

    def load(self, filename):
        # TODO: Carregar o modelo de um arquivo
        pass

    def log(self, message):
        # Simplesmente imprimir a mensagem para este exemplo
        print(message)
