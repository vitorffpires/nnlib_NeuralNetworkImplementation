from models.model import Model
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