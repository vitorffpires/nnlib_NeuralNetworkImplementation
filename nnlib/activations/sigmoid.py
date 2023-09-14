import numpy as np
from Activation import Activation
class Sigmoid(Activation):
    """
    Função de ativação Sigmoid.
    """

    def activate(self, input_data):
        """
        Calcula a ativação Sigmoid para os dados de entrada fornecidos.
        """
        return 1 / (1 + np.exp(-input_data))

    def derivate(self, d_output):
        """
        Calcula a derivada da função Sigmoid em relação à saída.
        """
        sigmoid = self.forward(d_output)
        return sigmoid * (1 - sigmoid)
       
if __name__ == '__main__':
    # Create an instance of the Linear activation function
    activation_function = Sigmoid()
    
    # Test data
    data = np.array([[-1.0, 1.0], [1.0, -1.0]])
    print("Input Data:", data)
    
    # Forward pass
    forward_result = activation_function.activate(data)
    print("Forward Result:", forward_result)
    
    # Backward pass
    backward_result = activation_function.derivate(data)
    print("Backward Result:", backward_result)