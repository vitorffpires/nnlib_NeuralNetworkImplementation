import numpy as np 
import joblib
from nnlib.layers.layer import Layer
from nnlib.loss_functions.loss import LossFunction
from nnlib.optimization_functions.optimizer import Optimizer
from nnlib.optimization_functions.adam import AdaptiveMomentEstimation
from nnlib.initialization_functions.initializer import Initializer
from nnlib.activation_functions.activation import Activation

class SequentialModel():
    
    def __init__(self) -> None:
        self.layers = []
        self.optimizer = None
        self.loss = None
        self.best_params = {'loss': float('inf'), 'weights': None, 'epoch': 0}
    
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
    
    def compile(self, optimizer: Optimizer, loss: LossFunction, initializer: Initializer, X: np.array = None) -> None:
        self.optimizer = optimizer
        self.loss = loss

        # Initialize weights if initializer is provided
        if initializer is not None:

            input_dim = self.layers[0].input_dim
    
            # If input_dim is not set, try to infer it from X
            if input_dim is None:
                if X is not None:
                    input_dim = X.shape[1]  # Assuming X is 2D: (n_samples, n_features)
                    self.layers[0].input_dim = input_dim  # Set input_dim for the first layer
                else:
                    raise ValueError("input_dim is not set for the first layer and cannot be inferred from X because X is None.")
            
            for layer in self.layers:
                # Initialize weights 
                weights = initializer.initialize_weights(input_dim, layer.n_units)
                if optimizer == AdaptiveMomentEstimation:
                    # Initialize m and v for Adam
                    layer.m = np.zeros_like(weights)
                    layer.v = np.zeros_like(weights)
    
                # Set the initialized weights to the layer
                layer.set_weights(weights)
    
                # Update input_dim for the next layer
                input_dim = layer.n_units
    
    def fit(self, X: np.array, y: np.array, epochs: int, batch_size: int, 
            X_val: np.array = None, y_val: np.array = None, verbose: bool = True) -> None:
        for epoch in range(epochs):
            epoch_losses = []  # To store loss for each batch in the epoch

            # Batch training
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                # Forward pass
                output = X_batch
                i=0
                for layer in self.layers:
                    i=i+1
                    output = layer.forward(output)

                # Compute loss
                loss_value = self.loss.compute(y_batch, output)
                epoch_losses.append(loss_value)
                #print(f'the loss is: {loss_value}')

                # Backward pass
                dLda = self.loss.derivate(y_batch, output)
                #print(f'on loss the derivate is: \n {dLda}')
                i=0
                for layer in reversed(self.layers):
                    i=i+1
                    dLda = layer.backward(dLda)
                    #print(f'on layer {i} the derivate on the backward is:')
                    #print(dLda)

                    # Update parameters
                    #print(f'pessos na camada {i} antes do otimizador:') 
                    #print(f'{layer.weights}')
                    self.optimizer.update(layer)
                    #print(f'pessos na camada {i} depois do otimizador:') 
                    #print(f'{layer.weights}')

            # Compute average loss for the epoch
            avg_epoch_loss = np.average(epoch_losses)

            # Validate the model if validation data is provided
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)

                # Check and update best parameters if current validation loss is lower
                if val_loss < self.best_params['loss']:
                    self.best_params['loss'] = val_loss
                    self.best_params['weights'] = [layer.get_weights() for layer in self.layers]
                    self.best_params['epoch'] = epoch

            # Implement logging
            if verbose:
                log_msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f}"
                if X_val is not None and y_val is not None:
                    log_msg += f" - val_loss: {val_loss:.4f}"
                print(log_msg)
        # Update layers with the best weights found during training
        for i, layer in enumerate(self.layers):
            layer.set_weights(self.best_params['weights'][i])    

    def evaluate(self, X: np.array, y: np.array) -> float:
        """
        Evaluate the model on the provided data.
        
        Parameters:
        - X: np.array
            Input data.
        - y: np.array
            True labels.
        
        Returns:
        float:
            Loss value.
        """
        # Forward pass
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        
        # Compute loss
        loss_value = self.loss.compute(y, output)
        return loss_value
    
    def predict(self, X: np.array) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def export_net(self, filename: str) -> None: 
         # Save the model to a file
        joblib.dump(self, filename)

    def import_net(filename: str) -> 'SequentialModel':
        # Load the model from a file
        return joblib.load(filename)
