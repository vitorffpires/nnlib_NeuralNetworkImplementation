import numpy as np

class Initializer:
    @staticmethod
    def he(shape, scale=2.0):
        """
        Inicialização de He, recomendada para ReLU e variantes.
        :param shape: Forma dos pesos a serem inicializados.
        :param scale: Fator de escala.
        :return: Array numpy inicializado.
        """
        stddev = np.sqrt(scale / shape[0])
        return np.random.normal(loc=0, scale=stddev, size=shape)

    @staticmethod
    def xavier(shape):
        """
        Inicialização Xavier (ou Glorot), recomendada para funções de ativação sigmóides e hiperbólicas.
        :param shape: Forma dos pesos a serem inicializados.
        :return: Array numpy inicializado.
        """
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)

    @staticmethod
    def normal(shape, mean=0, stddev=0.01):
        """
        Inicialização normal.
        :param shape: Forma dos pesos a serem inicializados.
        :param mean: Média da distribuição normal.
        :param stddev: Desvio padrão da distribuição normal.
        :return: Array numpy inicializado.
        """
        return np.random.normal(loc=mean, scale=stddev, size=shape)

    @staticmethod
    def uniform(shape, minval=-0.01, maxval=0.01):
        """
        Inicialização uniforme.
        :param shape: Forma dos pesos a serem inicializados.
        :param minval: Valor mínimo da distribuição uniforme.
        :param maxval: Valor máximo da distribuição uniforme.
        :return: Array numpy inicializado.
        """
        return np.random.uniform(minval, maxval, size=shape)

# Exemplo de uso:
# weights = Initializer.he(shape=(256, 512))
