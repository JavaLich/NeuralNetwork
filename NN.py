import numpy as np


def MSE(y, y_hat, m):
    return .5 * np.sum((y - y_hat) ** 2) / m


def MSE_prime(y, y_hat, m):
    return (y - y_hat) / m


def relu(z):
    if z <= 0:
        return 0
    return min(z, 1)


def relu_prime(z):
    if z <= 0:
        return 0
    return 1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


class Layer:

    def __init__(self, input_size, size):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.randn(self.input_size, self.size)
        self.bias = np.random.randn(self.size)
        self.bias = np.zeros(self.size)
        self.z = 0
        self.x = np.zeros(self.weights.shape)

    def evaluate(self, x):
        self.x = x
        self.z = np.dot(x, self.weights) + self.bias
        return sigmoid(self.z)


class NeuralNetwork:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

    def layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        a = x
        for i in self.layers:
            a = i.evaluate(a)
        return a

    def backprop(self, x, y, m):
        yHat = self.forward(x)
        cost = MSE(y, yHat, m)
        delta = np.multiply(-MSE_prime(y, yHat, m), sigmoidPrime(self.layers[-1].z))
        do = np.dot(self.layers[-1].x.T, delta)
        der = []
        db = []
        der.insert(0, do)
        db.insert(0, np.sum(delta, axis=0))
        for i in reversed(range(len(self.layers))):
            if self.layers[i] == self.layers[-1]:
                continue
            delta = np.dot(delta, self.layers[i + 1].weights.T) * sigmoidPrime(self.layers[i].z)
            der.insert(0, np.dot(self.layers[i].x.T, delta))
            db.insert(0, np.sum(delta, axis=0))
        return der, db, cost

    def train(self, lr, x, y, m, iterations):
        der, db, cost = self.backprop(x, y, m)
        print(cost)

        for j in range(iterations):
            for i in range(len(self.layers)):
                descent = lr * der[i]
                biasDescent = lr * db[i]
                self.layers[i].weights -= descent
                self.layers[i].bias -= biasDescent
            der, db, cost = self.backprop(x, y, m)
        print(cost)

    def test(self, x, y, iterations):
        accuracy = 0
        print(y)
        for j in range(iterations):
            index = int(np.random.rand(1) * len(x))
            test = x[index]
            a = self.forward(test)
            if a >= .5:
                a = 1
            else:
                a = 0

            print("Input: {}, Actual: {}, Desired: {}".format(test, [a], y[index]))
            if a == y[index]:
                accuracy += 1
        print("Accuracy: {}".format(accuracy / iterations))
