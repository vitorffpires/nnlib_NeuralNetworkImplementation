import numpy as np 
import pickle
from nnlib.layers.layer import Layer
from nnlib.optimization_functions.optimizer import Optimizer
from nnlib.loss_functions.loss import LossFunction
from nnlib.initialization_functions.initializer import Initializer

class SequentialModel():
    
    def __init__(self) -> None:
        self.layers = []
        self.optimizer = None
        self.loss = None
        self.best_params = {'loss': float('inf'), 'weights': None, 'bias': None, 'epoch': 0}
    
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
    
    def compile(self, optimizer: Optimizer, loss: LossFunction, initializer: Initializer) -> None:
        self.optimizer = optimizer
        self.loss = loss

        # Initialize weights if initializer is provided
        if initializer is not None:
            self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Initializer) -> None:
        input_dim = self.layers[0].input_dim  # Assuming input_dim for the first layer is set
        
        for layer in self.layers:
            # Initialize weights and biases
            weights = initializer.initialize_weights(input_dim, layer.n_units)
            biases = initializer.initialize_bias(layer.n_units)

            # Set the initialized weights and biases to the layer
            layer.set_weights({'weights': weights, 'biases': biases})

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
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                loss_value = self.loss.compute(y_batch, output)
                epoch_losses.append(loss_value)

                # Backward pass
                dLda = self.loss.derivative(y_batch, output)  # Corrected method name
                for layer in reversed(self.layers):
                    dLda = layer.backward(dLda)

                    # Update parameters
                    self.optimizer.update(layer)

            # Log epoch data
            avg_epoch_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_epoch_loss)

            # Validate the model if validation data is provided
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                self.history.setdefault('val_loss', []).append(val_loss)

                # Check and update best parameters if current validation loss is lower
                if val_loss < self.best_params['loss']:
                    self.best_params['loss'] = val_loss
                    self.best_params['weights'] = [layer.get_weights()['weights'] for layer in self.layers]
                    self.best_params['bias'] = [layer.get_weights()['biases'] for layer in self.layers]
                    self.best_params['epoch'] = epoch

            # Implement logging
            if verbose:
                log_msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f}"
                if X_val is not None and y_val is not None:
                    log_msg += f" - val_loss: {val_loss:.4f}"
                print(log_msg)

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

    def save_model(self, filename: str) -> None:
        """
        Save the model parameters to a file.

        Parameters:
        - filename: str
            The name of the file to save the model parameters.
        """
        model_params = {
            'weights': [layer.get_weights()['weights'] for layer in self.layers],
            'biases': [layer.get_weights()['biases'] for layer in self.layers],
            'best_params': self.best_params,
            'history': self.history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
    
    def load_model(self, filename: str) -> None:
        """
        Load the model parameters from a file.

        Parameters:
        - filename: str
            The name of the file from which to load the model parameters.
        """
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        
        for i, layer in enumerate(self.layers):
            layer.set_weights({
                'weights': model_params['weights'][i],
                'biases': model_params['biases'][i]
            })
        
        self.best_params = model_params['best_params']
        self.history = model_params['history']
    
    def predict(self, X: np.array) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
