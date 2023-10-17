import numpy as np
from nnlib.loss_functions.loss import LossFunction

class CategoricalCrossEntropy(LossFunction):
    """
    Implements the Categorical Cross-Entropy loss function for multi-class classification problems.
    
    Categorical Cross-Entropy is a loss function that measures the dissimilarity between the true labels 
    and the predicted probabilities. It is commonly used in training neural networks for multi-class 
    classification tasks.
    
    Methods:
    - compute(ypredict, ytrue): Computes the categorical cross-entropy loss.
    - derivate(ypredict, ytrue): Computes the gradient of the loss with respect to the predictions.
    
    Attributes:
    None
    """
    def compute(self, ypredict: np.array, ytrue: np.array) -> float:
        """
        Compute the categorical cross-entropy loss between true labels and predicted probabilities.
        
        Parameters:
        - ypredict: np.array
            Predicted probabilities from the model. Each row should sum to 1.
        - ytrue: np.array
            One-hot encoded true labels.
        
        Returns:
        float:
            Categorical cross-entropy loss value.
        """
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
            
        epsilon = 1e-7
        ypredict = np.clip(ypredict, epsilon, 1 - epsilon)
        
        return -np.mean(np.sum(ytrue * np.log(ypredict), axis=1, keepdims=True))
    
    
    def derivate(self, ypredict: np.array, ytrue: np.array) -> np.array:
        """
        Compute the gradient of the categorical cross-entropy loss with respect to the predictions.
        
        Parameters:
        - ypredict: np.array
            Predicted probabilities from the model. Each row should sum to 1.
        - ytrue: np.array
            One-hot encoded true labels.
        
        Returns:
        np.array:
            Gradient of the loss with respect to the predictions.
        """
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
        
        epsilon = 1e-7
        ypredict = np.clip(ypredict, epsilon, 1 - epsilon)
        return ypredict - ytrue
