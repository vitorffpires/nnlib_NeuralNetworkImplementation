import numpy as np
from Activation import Activation
class Linear(Activation):
    """
    Função de ativação Linear.
    """

    def activate(self, input_data):
        """
        Retorna os dados de entrada fornecidos (função identidade).

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Os próprios dados de entrada.
        """
        return input_data

    def derivate(self, d_output):
        """
        Calcula a derivada da função Linear em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função Linear.
        """
        return np.ones_like(d_output)

if __name__ == '__main__':
    # Create an instance of the Linear activation function
    activation_function = Linear()
    
    # Test data
    data = np.array([[-1.0, 1.0], [1.0, -1.0]])
    print("Input Data:", data)
    
    # Forward pass
    forward_result = activation_function.activate(data)
    print("Forward Result:", forward_result)
    
    # Backward pass
    backward_result = activation_function.derivate(data)
    print("Backward Result:", backward_result)