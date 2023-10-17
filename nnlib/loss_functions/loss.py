from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
     
    @abstractmethod
    def compute(self, ytrue: np.array, ypredict: np.array) -> float:
        pass


    @abstractmethod
    def derivate(self, ytrue: np.array, ypredict: np.array) -> np.array:
        pass