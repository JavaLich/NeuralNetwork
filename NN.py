import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Layer:

    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.randn(self.input_size, self.size)
        self.bias = np.random.randn(self.input_size, self.size)

    def evaluate(self, x):
        z2 = np.dot(x, self.weights) + self.bias
        return sigmoid(z2)


class NeuralNetwork:

    def __init__(self, input_size, output_size, layers):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
