from callback import Callback
class ModelCheckPoint(Callback):
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
        # TODO model.save(self.filepath.format(epoch=epoch))
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass