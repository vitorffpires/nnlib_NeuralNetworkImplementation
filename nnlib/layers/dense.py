from nnlib.layers.layer import Layer
import numpy as np

class Dense(Layer):
    """
    Dense (fully connected) layer.
    
    Inherits from Layer.
    """
    def forward(self, input: np.array) -> np.array:
        """
        Forward pass for the dense layer.
        
        Parameters:
        - input (np.array): Input data.
        
        Returns:
        np.array:
            Output after forward pass.
        """
        self.layer_input = input
        self.weighted_vector = np.dot(input, self.weights) + self.bias
        self.layer_output = self.activation.activate(self.weighted_vector) 

        return self.layer_output 
    
    
    def backward(self, loss_derivative: np.array) -> np.array:
        """
        Backward pass for the dense layer.
        
        Parameters:
        - loss_derivative (np.array): Derivative of the loss.
        
        Returns:
        np.array:
            Gradient of the loss with respect to the input.
        """
        # Calculating the derivative of the activation function
        activation_derivative = self.activation.derivate(x = self.weighted_vector)
        
        self.delta = np.multiply(loss_derivative, activation_derivative) 
        
        # Calculating the gradient of the weights
        self.derivative_weights = np.dot(self.layer_input.T, self.delta)
        
        # Calculating the gradient of the bias
        self.derivative_bias = np.sum(self.delta, axis=0, keepdims=True)
        
        # Calculating the gradient of the loss with respect to the previous layer's activations
        l1_loss_derivative = np.dot(self.delta, self.weights.T) 
        
        return l1_loss_derivative
    
    
    def get_weights(self) -> dict:
        """
        Get the weights of the dense layer.
        
        Returns:
        dict:
            Weights of the layer.
        """
        return {"weights": self.weights, "bias": self.bias}
    
    
    def set_weights(self, weights: dict) -> None:
        """
        Set the weights of the dense layer.
        
        Parameters:
        - weights (dict): Weights to be set.
        """
        self.weights = weights["weights"]
        self.bias = weights["bias"]
