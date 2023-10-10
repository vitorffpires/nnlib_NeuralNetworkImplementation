from nnlib.layers.layer import Layer
from abc import ABC, abstractmethod

class Optimizer(ABC):
    
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
    
        
    @abstractmethod
    def update(self, layer: Layer) -> None:
        pass
    
    
    def backpropagate(self, layers: list) -> None:
        for layer in layers:
            self.update(layer)