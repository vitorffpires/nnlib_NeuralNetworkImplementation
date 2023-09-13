import numpy as np
import matplotlib.pyplot as plt

class Visualization:

    @staticmethod
    def plot_loss(train_loss, val_loss=None):
        """
        Visualizar a perda ao longo do treinamento.
        :param train_loss: Lista de perda de treinamento ao longo das épocas.
        :param val_loss: (Opcional) Lista de perda de validação ao longo das épocas.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Train Loss", color="blue")
        
        if val_loss is not None:
            plt.plot(val_loss, label="Validation Loss", color="red")
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over epochs")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_weights(weights):
        """
        Mostrar a distribuição dos pesos.
        :param weights: Lista ou array de pesos.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=30, color="blue", alpha=0.7)
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.title("Distribution of weights")
        plt.grid(True)
        plt.show()

# Exemplo de uso:
# Visualization.plot_loss([0.9, 0.7, 0.5, 0.3], [0.85, 0.65, 0.48, 0.29])
# Visualization.plot_weights(np.random.randn(1000))
