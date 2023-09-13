import numpy as np

class DataLoader:
    def __init__(self, dataset, labels, batch_size=32, shuffle=True, transform=None):
        """
        Inicializa o data loader.

        Args:
        - dataset (np.ndarray): Os dados a serem carregados.
        - labels (np.ndarray): Rótulos correspondentes ao conjunto de dados.
        - batch_size (int): Tamanho do lote.
        - shuffle (bool): Se os dados devem ser embaralhados antes de serem servidos.
        - transform (callable, optional): Uma função de transformação a ser aplicada aos dados.
        """
        self.dataset = dataset
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """
        Retorna o número de lotes.
        """
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        """
        Retorna um iterador para os lotes.
        """
        for start_idx in range(0, len(self.dataset) - self.batch_size + 1, self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_indices = self.indices[start_idx:end_idx]
            X = self.dataset[batch_indices]
            y = self.labels[batch_indices]
            if self.transform:
                X = self.transform(X)
            yield X, y

# Exemplo de uso:
# Suponha que data e labels são seus dados e rótulos respectivos
# data = np.random.randn(100, 32, 32, 3)  # 100 imagens RGB 32x32
# labels = np.random.randint(0, 2, 100)  # 100 rótulos (0 ou 1)

# loader = DataLoader(data, labels, batch_size=10, shuffle=True)
# for batch_data, batch_labels in loader:
#     pass  # Aqui você pode usar batch_data e batch_labels para treinamento/avaliação
