import numpy as np
from optmizer import Optimizer
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