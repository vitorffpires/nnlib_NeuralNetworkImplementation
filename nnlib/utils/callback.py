from abc import ABC, abstractmethod
import numpy as np

class Callback(ABC):
    """
    Classe base para todos os callbacks.
    """

    def __init__(self):
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch, logs=None):
        """
        Chamado no início de cada época.
        """
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, logs=None):
        """
        Chamado no final de cada época.
        """
        pass

    @abstractmethod
    def on_batch_begin(self, batch, logs=None):
        """
        Chamado no início de cada lote.
        """
        pass

    @abstractmethod
    def on_batch_end(self, batch, logs=None):
        """
        Chamado no final de cada lote.
        """
        pass