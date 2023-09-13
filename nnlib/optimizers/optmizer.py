import numpy as np

class Optimizer:
    """
    Classe base para todos os otimizadores.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gradient):
        """
        Atualiza os pesos com base no gradiente.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Otimizador Stochastic Gradient Descent (SGD).
    """

    def update(self, gradient):
        """
        Atualiza os pesos usando o método SGD.
        """
        return -self.learning_rate * gradient


class Adam(Optimizer):
    """
    Otimizador Adam.
    """
    
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, gradient):
        """
        Atualiza os pesos usando o método Adam.
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        
        m_corr = self.m / (1 - self.beta1**self.t)
        v_corr = self.v / (1 - self.beta2**self.t)
        
        return -self.learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)


class Regularization:
    """
    Classe base para regularizações. Esta não é um otimizador, mas pode ser usado junto com um.
    """

    def __init__(self, lambda_value):
        self.lambda_value = lambda_value

    def regularization_term(self, weights):
        """
        Calcula o termo de regularização.
        """
        raise NotImplementedError


class L2Regularization(Regularization):
    """
    Regularização L2.
    """

    def regularization_term(self, weights):
        """
        Calcula o termo de regularização L2.
        """
        return self.lambda_value * np.sum(weights**2)