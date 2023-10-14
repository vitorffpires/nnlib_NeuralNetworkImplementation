from nnlib.loss_functions.loss import LossFunction
import numpy as np

class MeanSquaredError(LossFunction):

    def compute(self, ypredict: np.array, ytrue: np.array) -> float:
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
        loss = np.mean((ytrue - ypredict)**2)
        return loss

        
    def derivate(self, ypredict: np.array, ytrue: np.array) -> np.array:
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
        gradient = 2 * (ytrue - ypredict)
        return gradient