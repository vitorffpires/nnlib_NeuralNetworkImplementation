import numpy as np
from Activation import Activation
class Softmax(Activation):
    """
    Função de ativação Softmax.
    """

    def activate(self, input_data):
        """
        Calcula a ativação Softmax para os dados de entrada fornecidos.

        Parameters:
        - input_data: numpy.ndarray
            Dados de entrada para os quais a ativação será calculada.

        Returns:
        numpy.ndarray:
            Valores de ativação calculados usando Softmax.
        """
        exp_data = np.exp(input_data - np.max(input_data))
        return exp_data / np.sum(exp_data, axis=-1, keepdims=True)

    def derivate(self, d_output):
        """
        Calcula a derivada da função Softmax em relação à saída.

        Parameters:
        - d_output: numpy.ndarray
            Gradiente da camada seguinte.

        Returns:
        numpy.ndarray:
            Gradiente da função Softmax.
        """
        s = self.forward(d_output)
        return s * (1 - s)
    
if __name__ == '__main__':
    # Create an instance of the Linear activation function
    activation_function = Softmax()
    
    # Test data
    data = np.array([[-1.0, 1.0], [1.0, -1.0]])
    print("Input Data:", data)
    
    # Forward pass
    forward_result = activation_function.activate(data)
    print("Forward Result:", forward_result)
    
    # Backward pass
    backward_result = activation_function.derivate(data)
    print("Backward Result:", backward_result)