import numpy as np
from nnlib.loss_functions.loss import LossFunction

class CategoricalCrossEntropy(LossFunction):
        
    
    def compute(self, ypredict: np.array, ytrue: np.array) -> float:
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
            
        epsilon = 1e-7
        ypredict = np.clip(ypredict, epsilon, 1 - epsilon)
        
        return -np.mean(np.sum(ytrue * np.log(ypredict), axis=1, keepdims=True))
    
    
    def derivate(self, ypredict: np.array, ytrue: np.array) -> np.array:
        if (ypredict.shape != ytrue.shape) or (ypredict.ndim != 2) or (ytrue.ndim != 2):
            raise ValueError(f"for the loss, both ypredict and y_true must be of the same dimension and 2D arrays, got {ypredict.shape} and {ytrue.shape}")
        
        epsilon = 1e-7
        ypredict = np.clip(ypredict, epsilon, 1 - epsilon)
        return ypredict - ytrue
