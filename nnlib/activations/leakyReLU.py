import numpy as np
from Activation import Activation
class LeakyReLU(Activation):
    """
    Função de ativação Leaky ReLU.
    """
    def __init__(self, alpha=0.01):
        """
        Construtor da função LeakyReLU.

        Parameters:
        - alpha: float
            Coeficiente que determina o valor "leaky".
        """
        super().__init__()
        self.alpha = alpha

    def activate(self, input_data):
        """
        Calcula a ativação LeakyReLU para os dados de entrada fornecidos.

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Valores de ativação calculados usando LeakyReLU.
        """
        return np.where(input_data > 0, input_data, self.alpha * input_data)

    def derivate(self, d_output):
        """
        Calcula a derivada da função LeakyReLU em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função LeakyReLU.
        """
        return np.where(d_output > 0, 1, self.alpha)
    
if __name__ == '__main__':
    # Create an instance of the Linear activation function
    activation_function = LeakyReLU(0.5)
    
    # Test data
    data = np.array([[-1.0, 1.0], [1.0, -1.0]])
    print("Input Data:", data)
    
    # Forward pass
    forward_result = activation_function.activate(data)
    print("Forward Result:", forward_result)
    
    # Backward pass
    backward_result = activation_function.derivate(data)
    print("Backward Result:", backward_result)