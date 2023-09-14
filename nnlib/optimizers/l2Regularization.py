import numpy as np
from regularization import Regularization
class L2Regularization(Regularization):
    """
    Regularização L2.
    """

    def regularization_term(self, weights):
        """
        Calcula o termo de regularização L2.
        """
        return self.lambda_value * np.sum(weights**2)