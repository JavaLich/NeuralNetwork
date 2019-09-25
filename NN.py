import numpy as np
import matplotlib.pyplot as plt


def MSE(y, y_hat):
    return .5 * np.sum((y - y_hat) ** 2)


def MSE_prime(y, y_hat):
    return y - y_hat


def binary_cross_entropy(y, y_hat):
    return -np.sum((y*np.log(y_hat)+(1-y)*np.log(1-y_hat)))


def binary_cross_entropy_prime(y, p):
    return (y / p) - ((1 - y) / (1 - p))


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return np.where(z > 0, 1.0, 0.0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def linear(z):
    return z


def linearPrime(z):
    return np.ones(z.shape)


funcDict = {
    "sigmoid": sigmoid,
    "sigmoidPrime": sigmoidPrime,
    "relu": relu,
    "linear": linear,
    "linearPrime": linearPrime,
    "reluPrime": relu_prime,
    "MSE": MSE,
    "MSEPrime": MSE_prime,
    "binaryCrossEntropy": binary_cross_entropy,
    "binaryCrossEntropyPrime": binary_cross_entropy_prime
}


class Layer:

    def __init__(self, input_size, size, act_func="sigmoid"):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.rand(self.input_size, self.size) * np.sqrt(2/size)
        self.bias = np.zeros(self.size)
        self.z = 0
        self.x = np.zeros(self.weights.shape)
        self.act_func = act_func

    def evaluate(self, x):
        self.x = x
        self.z = np.dot(x, self.weights) + self.bias
        return funcDict[self.act_func](self.z)


def create_mini_batches(x, y, batch_size):
    mini_batches_x = []
    mini_batches_y = []
    for i in range(0, len(x), batch_size):
        mini_batches_x.append(x[i:i + batch_size])
        mini_batches_y.append(y[i:i + batch_size])

    return mini_batches_x, mini_batches_y


# Did you look for jobs in physics and what type of jobs did you see
# Do you think Computer Science and Physics can complement each other well
# Would double majoring in CS/Physics be too hard


class NeuralNetwork:

    def __init__(self, input_size, output_size, cost_func="MSE"):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.cost_func = cost_func

    def layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        a = x
        for i in self.layers:
            a = i.evaluate(a)
        return a

    def backprop(self, x, y):
        yHat = self.forward(x)
        cost = funcDict[self.cost_func](y, yHat)
        delta = np.multiply(-funcDict[self.cost_func + "Prime"](y, yHat),
                            funcDict[self.layers[-1].act_func + "Prime"](self.layers[-1].z))
        do = np.dot(self.layers[-1].x.T, delta)
        der = []
        db = []
        der.insert(0, do)
        db.insert(0, np.sum(delta, axis=0))
        for i in reversed(range(len(self.layers))):
            if self.layers[i] == self.layers[-1]:
                continue
            delta = np.dot(delta, self.layers[i + 1].weights.T) * funcDict[self.layers[i].act_func + "Prime"](
                self.layers[i].z)
            der.insert(0, np.dot(self.layers[i].x.T, delta))
            db.insert(0, np.sum(delta, axis=0))
        return der, db, cost, yHat

    def train(self, lr, x, y, batch_size=32, epochs=1):
        costs = []

        for itr in range(epochs):
            mini_batches_x, mini_batches_y = create_mini_batches(x, y, batch_size)

            np.random.shuffle(mini_batches_x)
            np.random.shuffle(mini_batches_y)
            for batch in range(len(mini_batches_x)):
                der, db, cost, yHat = self.backprop(mini_batches_x[batch], mini_batches_y[batch])

                for i in range(len(self.layers)):
                    descent = lr * der[i]
                    biasDescent = lr * db[i]
                    self.layers[i].weights -= descent
                    self.layers[i].bias -= biasDescent
                costs.append(MSE(mini_batches_y[batch], yHat))
        plt.plot(costs)
        plt.show()

    def print(self):
        for i in range(len(self.layers)):
            print(self.layers[i].weights)
            print(self.layers[i].bias)

    def printEval(self, x):
        a = x
        for i in self.layers:
            a = i.evaluate(a)
            print(a)

    def test(self, x, y, iterations):
        accuracy = 0
        for j in range(iterations):
            index = j % len(x)
            test = x[index]
            a = self.forward(test)
            print("Input: {}, Actual: {}, Desired: {}".format(test, a, y[index]))

            accuracy += np.sqrt(MSE(y[index], a))
        print("Accuracy: {}".format((accuracy / iterations)))
