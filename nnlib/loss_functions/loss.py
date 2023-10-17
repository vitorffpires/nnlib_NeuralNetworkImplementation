from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    """
    Abstract base class for loss functions.
    """
    @abstractmethod
    def compute(self, ytrue: np.array, ypredict: np.array) -> float:
        """
        Compute the loss value given the true labels and predicted labels.
        
        Parameters:
        - ytrue (np.array): True labels.
        - ypredict (np.array): Predicted labels.
        
        Returns:
        float:
            Computed loss value.
        """
        pass


    @abstractmethod
    def derivate(self, ytrue: np.array, ypredict: np.array) -> np.array:
        """
        Compute the derivative of the loss function with respect to the predicted labels.
        
        Parameters:
        - ytrue (np.array): True labels.
        - ypredict (np.array): Predicted labels.
        
        Returns:
        np.array:
            Derivative of the loss.
        """
        pass