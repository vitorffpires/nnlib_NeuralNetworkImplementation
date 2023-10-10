from nnlib.layers.layer import Layer
import numpy as np

class Dense(Layer):
    
    def forward(self, input: np.array) -> np.array:
        self.input_vector = input
        self.weighted_vector = np.dot(input, self.weights) + self.biases 
        self.output_vector = self.activation.activate(self.weighted_vector) 
        
        return self.output_vector 
    
    
    def backward(self, loss_derivative: np.array) -> np.array:
        # Calculating the derivative of the activation function
        activation_derivative = self.activation.derivate(self.weighted_vector) 
        
        self.delta = np.multiply(loss_derivative, activation_derivative) 
        
        # Calculating the gradient of the weights and biases
        self.derivative_weights = np.dot(self.input_vector.T, self.delta)  
        self.derivative_biases = np.sum(self.delta, axis=0, keepdims=True) if self.has_bias else None 
        
        # Calculating the gradient of the loss with respect to the previous layer's activations
        loss_derivative_prev = np.dot(self.delta, self.weights.T) 
        
        return loss_derivative_prev
    
    
    def get_weights(self) -> dict:
        return {'weights': self.weights, 'biases': self.biases}
    
    
    def set_weights(self, weights: dict) -> None:
        self.weights = weights['weights']
        self.biases = weights['biases']
