from nnlib.layers.layer import Layer
from nnlib.activation_functions.softmax import Softmax
import numpy as np

class Dense(Layer):
    
    def forward(self, input: np.array) -> np.array:
        self.layer_input = input
        self.weighted_vector = np.dot(input, self.weights) 
        self.layer_output = self.activation.activate(self.weighted_vector) 
        
        return self.layer_output 
    
    
    def backward(self, loss_derivative: np.array) -> np.array:
        # Calculating the derivative of the activation function
        activation_derivative = self.activation.derivate(x = self.weighted_vector)
        
        self.delta = np.multiply(loss_derivative, activation_derivative) 
        
        # Calculating the gradient of the weights
        self.derivative_weights = np.dot(self.layer_input.T, self.delta)
        
        # Calculating the gradient of the loss with respect to the previous layer's activations
        l1_loss_derivative = np.dot(self.delta, self.weights.T) 
        
        return l1_loss_derivative
    
    
    def get_weights(self) -> dict:
        return self.weights
    
    
    def set_weights(self, weights: np.array) -> None:
        self.weights = weights
