import numpy as np
from nnlib.Layers import layer
from nnlib.Activations import sigmoid
class Dense(layer):
    """
    Camada Densa (ou Totalmente Conectada).
    """

    def __init__(self, input_dim, output_dim, activation_function):
        # TODO mapping for activation function
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.activation = activation_function

    def forward_pass(self, input_data):
        self.input_data = input_data
        self.output = self.activation.activate(np.dot(input_data, self.weights) + self.bias)
        return self.output

    def backward_pass(self, output_grad):
        d_activation = self.activation.derivate(self.output) * output_grad
        d_weights = np.dot(self.input_data.T, d_activation)
        d_bias = np.sum(d_activation, axis=0, keepdims=True)
        d_input_data = np.dot(d_activation, self.weights.T)
        
        # Atualização dos pesos e bias será realizada pelo otimizador
        self.d_weights = d_weights
        self.d_bias = d_bias

        return d_input_data
    

if __name__ == '__main__':
    # Sample data
    data = np.array([[1.0, -1.0], [-1.0, 1.0]])
    activation = sigmoid()
    
    # Instantiate Dense layer with LeakyReLU activation
    dense_layer = Dense(2, 3, activation)
    
    # Forward pass
    output = dense_layer.forward_pass(data)
    print('Forward Pass Output:', output)
    
    # Backward pass
    gradient = dense_layer.backward_pass(output)#np.array([[1, 1, 1], [1, 1, 1]]))
    print('Backward Pass Gradient:', gradient)