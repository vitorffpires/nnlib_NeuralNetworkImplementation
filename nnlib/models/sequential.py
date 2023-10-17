import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from nnlib.layers.layer import Layer
from nnlib.loss_functions.loss import LossFunction
from nnlib.loss_functions.bce import BinaryCrossEntropy
from nnlib.loss_functions.mse import MeanSquaredError
from nnlib.optimization_functions.optimizer import Optimizer
from nnlib.optimization_functions.adam import AdaptiveMomentEstimation
from nnlib.initialization_functions.initializer import Initializer
from nnlib.activation_functions.activation import Activation

class SequentialModel():
    """
    A simple sequential neural network model.
    
    Attributes:
    - layers (list): A list of layers added to the model.
    - optimizer (Optimizer): The optimization algorithm used for training.
    - loss (LossFunction): The loss function used for training.
    - best_params (dict): Dictionary to store the best parameters during training.
    """
    
    def __init__(self) -> None:
        """Initialize the SequentialModel with empty layers and no optimizer or loss function."""
        self.layers = []
        self.optimizer = None
        self.loss = None
        self.best_params = {'loss': float('inf'), 'weights': None, 'epoch': 0}
    
    def add(self, layer: Layer) -> None:
        """
        Add a layer to the model.
        
        Parameters:
        - layer (Layer): The layer to be added.
        """
        self.layers.append(layer)
    
    def compile(self, optimizer: Optimizer, loss: LossFunction, initializer: Initializer, X: np.array = None) -> None:
        """
        Configure the model for training.
        
        Parameters:
        - optimizer (Optimizer): The optimization algorithm.
        - loss (LossFunction): The loss function.
        - initializer (Initializer): The weight initializer.
        - X (np.array, optional): Input data, used to infer input_dim if not set in the first layer.
        """
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
                weights = initializer.initialize_weights(input_dim = input_dim, n_units = layer.n_units)
                bias = np.zeros((1, layer.n_units))
                if isinstance(optimizer, AdaptiveMomentEstimation):
                    # Initialize m and v for Adam
                    layer.m = np.zeros_like(weights)
                    layer.v = np.zeros_like(weights)

                weights = {"weights": weights, "bias": bias}
                # Set the initialized weights to the layer
                layer.set_weights(weights = weights)
    
                # Update input_dim for the next layer
                input_dim = layer.n_units
    
    def fit(self, X: np.array, y: np.array, epochs: int, batch_size: int, 
            X_val: np.array = None, y_val: np.array = None, verbose: bool = True) -> None:
        """
        Train the model for a fixed number of epochs.
        
        Parameters:
        - X (np.array): Training data.
        - y (np.array): Target values.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Number of samples per gradient update.
        - X_val (np.array, optional): Validation data.
        - y_val (np.array, optional): Target values for validation data.
        - verbose (bool, optional): Whether to print training progress or not.
        """
        # Initialize lists to store metrics
        epoch_gradients = []
        epoch_activations = []
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for epoch in range(epochs):
            epoch_losses = []

            # Batch training
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                # Forward pass
                output = X_batch
                i=0
                for layer in self.layers:
                    i=i+1
                    output = layer.forward(input = output)
                #print(f'output da rede foi \n {output}')

                # Compute loss
                loss_value = self.loss.compute(ytrue = y_batch, ypredict = output)
                #print(f'o custo foi: {loss_value}')

                # Backward pass
                dLda = self.loss.derivate(ytrue = y_batch, ypredict =output)    
                epoch_losses.append(loss_value)
                #print(f'on loss the derivate is: \n {dLda}')
                i=0
                for layer in reversed(self.layers):
                    i=i+1
                    dLda = layer.backward(loss_derivative = dLda)

                    #print(f'on layer {i} the derivate on the backward is:')
                    #print(dLda)

                    # Update parameters
                    #print(f'pessos na camada {i} antes do otimizador:') 
                    #print(f'{layer.weights}')
                    self.optimizer.update(layer = layer)
                    #print(f'pessos na camada {i} depois do otimizador:') 
                    #print(f'{layer.weights}')
            # Compute average loss for the epoch
            for layer in self.layers:
                epoch_gradients.append(layer.derivative_weights)
                epoch_activations.append(layer.layer_output)
            avg_epoch_loss = np.average(epoch_losses)
            train_losses.append(avg_epoch_loss)
            train_pred = self.predict(X)
            if isinstance(self.loss, BinaryCrossEntropy):
                train_accuracy = accuracy_score(y, np.round(train_pred))
                train_accuracies.append(train_accuracy)
            elif isinstance(self.loss, MeanSquaredError):
                train_accuracy = mean_squared_error(y, np.round(train_pred))
                train_accuracies.append(train_accuracy)
            
            # Validate the model if validation data is provided
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X = X_val, y = y_val)
                val_losses.append(val_loss)
                val_pred = self.predict(X_val)
                if isinstance(self.loss, BinaryCrossEntropy):
                    val_accuracy = accuracy_score(y_val, np.round(val_pred))
                    val_accuracies.append(val_accuracy)
                elif isinstance(self.loss, MeanSquaredError):
                    val_accuracy = mean_squared_error(y_val, np.round(val_pred))
                    val_accuracies.append(val_accuracy)

                #print(f'val_loss foi: {val_loss}')

                # Check and update best parameters if current validation loss is lower
                if val_loss < self.best_params['loss']:
                    self.best_params['loss'] = val_loss
                    self.best_params['weights'] = [layer.get_weights() for layer in self.layers]
                    self.best_params['epoch'] = epoch+1

            # Implement logging
            if verbose:
                log_msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f}"
                if X_val is not None and y_val is not None:
                    log_msg += f" - val_loss: {val_loss:.4f}"
                print(log_msg)

        # Plotting for each epoch
        for epoch_idx in range(epochs):
            # Activations Histogram
            plt.figure(figsize=(15, 5))
            for i, activations in enumerate(epoch_activations[epoch_idx*len(self.layers):(epoch_idx*len(self.layers))+len(self.layers)]):
                plt.subplot(1, len(self.layers), i+1)
                plt.hist(activations, bins=8)
                plt.title(f"Layer {i+1} Activations Histogram - Epoch {epoch_idx+1}")
                plt.xlabel("Activation Value")
                plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

            # Gradients Histogram
            plt.figure(figsize=(15, 5))
            for i, gradients in enumerate(epoch_gradients[epoch_idx*len(self.layers):(epoch_idx*len(self.layers))+len(self.layers)]):
                plt.subplot(1, len(self.layers), i+1)
                plt.hist(gradients, bins=8)
                plt.title(f"Layer {i+1} Gradients Histogram - Epoch {epoch_idx+1}")
                plt.xlabel("Gradient Value")
                plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

        # Losses Line Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Training Loss", color="blue")
        if X_val is not None and y_val is not None:
            plt.plot(val_losses, label="Validation Loss", color="red")
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()

        # Accuracies Line Plot
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label="Training Accuracy", color="blue")
        if X_val is not None and y_val is not None:
            plt.plot(val_accuracies, label="Validation Accuracy", color="red")
        plt.title("Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy Value")
        plt.legend()
        plt.tight_layout()
        plt.show()



    def evaluate(self, X: np.array, y: np.array) -> float:
        """
        Evaluate the model on the provided data.
        
        Parameters:
        - X (np.array): Input data.
        - y (np.array): True labels.
        
        Returns:
        - float: Loss value.
        """
        # Forward pass
        output = X
        for layer in self.layers:
            output = layer.forward(input = output)
        #print(f'output da rede foi \n {output}')
        # Compute loss
        loss_value = self.loss.compute(ytrue = y, ypredict= output)
        return loss_value
    
    def predict(self, X: np.array) -> np.array:
        """
        Generate output predictions for the input samples.
        
        Parameters:
        - X (np.array): Input data.
        
        Returns:
        - np.array: Predictions.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(input = output)
        return output
    
    def export_net(self, filename: str) -> None: 
        """
        Save the model to a file.
        
        Parameters:
        - filename (str): The name of the file to save the model.
        """
        joblib.dump(self, filename)

    def import_net(filename: str) -> 'SequentialModel':
        """
        Load the model from a file.
        
        Parameters:
        - filename (str): The name of the file from which to load the model.
        
        Returns:
        - SequentialModel: The loaded model.
        """
        return joblib.load(filename)
