import numpy as np
from Activation import Activation
class ReLU(Activation):
    """
    Função de ativação ReLU (Rectified Linear Unit).
    """

    def activate(self, input_data):
        """
        Calcula a ativação ReLU para os dados de entrada fornecidos.
        """
        return np.maximum(0, input_data)

    def derivate(self, d_output):
        """
        Calcula a derivada da função ReLU em relação à saída.
        """
        return np.where(d_output > 0, 1, 0)
    
if __name__ == '__main__':
    # Create an instance of the Linear activation function
    activation_function = ReLU()
    
    # Test data
    data = np.array([[-1.0, 1.0], [1.0, -1.0]])
    print("Input Data:", data)
    
    # Forward pass
    forward_result = activation_function.activate(data)
    print("Forward Result:", forward_result)
    
    # Backward pass
    backward_result = activation_function.derivate(data)
    print("Backward Result:", backward_result)