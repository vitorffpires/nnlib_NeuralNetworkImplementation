from nnlib.loss_functions.loss import LossFunction
import numpy as np

class MeanSquaredError(LossFunction):

    def compute(self, ypredict: np.array, ytrue: np.array) -> float:
        loss = np.mean((ytrue - ypredict)**2)
        return loss

        
    def derivate(self, ypredict: np.array, ytrue: np.array) -> np.array:
        gradient = -2 * (ytrue - ypredict)
        return gradient