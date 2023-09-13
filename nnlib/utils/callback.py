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

class ModelCheckpoint(Callback):
    """
    Salva o modelo após cada época.
    """

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # Supondo que o modelo tenha um método 'save'
        model.save(self.filepath.format(epoch=epoch))

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
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
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                model.stop_training = True

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass
