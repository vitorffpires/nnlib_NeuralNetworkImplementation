from nnlib.loss_functions.loss import LossFunction
import numpy as np

class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross Entropy loss function.
    """
    def compute(self, ypredict: np.array, ytrue: np.array) -> float:
        """
        Compute the Binary Cross Entropy loss given the true labels and predicted labels.
        
        Parameters:
        - ytrue (np.array): True labels.
        - ypredict (np.array): Predicted labels.
        
        Returns:
        float:
            Computed Binary Cross Entropy loss value.
        """
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
        epsilon = 1e-15  # To prevent log(0)
        ypredict = np.clip(ypredict, epsilon, 1. - epsilon)  # Clip predictions
        return -np.mean(ytrue * np.log(ypredict) + (1 - ytrue) * np.log(1 - ypredict))
        

    def derivate(self, ypredict: np.array, ytrue: np.array) -> np.array:
        """
        Compute the derivative of the Binary Cross Entropy loss with respect to the predicted labels.
        
        Parameters:
        - ytrue (np.array): True labels.
        - ypredict (np.array): Predicted labels.
        
        Returns:
        np.array:
            Derivative of the Binary Cross Entropy loss.
        """
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
        epsilon = 1e-15  # To prevent division by zero
        ypredict = np.clip(ypredict, epsilon, 1. - epsilon)  # Clip predictions
        return -ytrue / ypredict + (1 - ytrue) / (1 - ypredict)