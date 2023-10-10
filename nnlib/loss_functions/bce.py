from nnlib.loss_functions.loss import LossFunction
import numpy as np

class BinaryCrossEntropyLoss(LossFunction):

    def compute(self, ypredict: np.array, ytrue: np.array) -> float:

        epsilon = 1e-15  # To prevent log(0)
        ypredict = np.clip(ypredict, epsilon, 1. - epsilon)  # Clip predictions
        loss = -np.mean(ytrue * np.log(ypredict) + (1 - ytrue) * np.log(1 - ypredict))
        return loss

    def derivate(self, ypredict: np.array, ytrue: np.array) -> np.array:

        epsilon = 1e-15  # To prevent division by zero
        ypredict = np.clip(ypredict, epsilon, 1. - epsilon)  # Clip predictions
        gradient = -(ytrue / ypredict - (1 - ytrue) / (1 - ypredict))
        return gradient