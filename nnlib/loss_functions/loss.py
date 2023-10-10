from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
     
    @abstractmethod
    def compute(self, ypredict: np.array, ytrue: np.array) -> float:
        pass


    @abstractmethod
    def derivate(self, ypredict: np.array, ytrue: np.array) -> np.array:
        pass