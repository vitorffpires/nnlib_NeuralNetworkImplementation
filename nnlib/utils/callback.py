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

class EarlyStopping(Callback):
    """
    Para o treinamento se a perda não melhorar após um determinado número de épocas.
    """

    def __init__(self, patience=10):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # TODO current_loss = logs.get('loss')
        # TODO if current_loss < self.best_loss:
        # TODO     self.best_loss = current_loss
        # TODO     self.wait = 0
        # TODO else:
        # TODO     self.wait += 1
        # TODO     if self.wait >= self.patience:
        # TODO         model.stop_training = True
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass
