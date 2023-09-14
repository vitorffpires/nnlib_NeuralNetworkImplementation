from optmizer import Optimizer
class SGD(Optimizer):
    """
    Otimizador Stochastic Gradient Descent (SGD).
    """

    def update(self, gradient):
        """
        Atualiza os pesos usando o método SGD.
        """
        return -self.learning_rate * gradient