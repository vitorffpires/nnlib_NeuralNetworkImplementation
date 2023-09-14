import numpy as np
from Activation import Activation
class Tanh(Activation):
    """
    Função de ativação Tanh (tangente hiperbólica).
    """

    def activate(self, input_data):
        """
        Calcula a ativação Tanh para os dados de entrada fornecidos.

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Valores de ativação calculados usando Tanh.
        """
        return np.tanh(input_data)

    def derivate(self, d_output):
        """
        Calcula a derivada da função Tanh em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função Tanh.
        """
        tanh_val = self.forward(d_output)
        return 1.0 - tanh_val**2
    
if __name__ == '__main__':
    # Create an instance of the Linear activation function
    activation_function = Tanh()
    
    # Test data
    data = np.array([[-1.0, 1.0], [1.0, -1.0]])
    print("Input Data:", data)
    
    # Forward pass
    forward_result = activation_function.activate(data)
    print("Forward Result:", forward_result)
    
    # Backward pass
    backward_result = activation_function.derivate(data)
    print("Backward Result:", backward_result)