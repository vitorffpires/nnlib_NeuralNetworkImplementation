from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Classe base para todas as camadas.
    """

    def __init__(self):
        pass

    @abstractmethod
    def forward_pass(self, input_data):
        pass

    @abstractmethod
    def backward_pass(self, output_grad):
        pass