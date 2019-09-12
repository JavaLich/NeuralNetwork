import numpy as np
import matplotlib.pyplot as plt


def MSE(y, y_hat, m):
    return .5 * np.sum((y - y_hat) ** 2) / m


def MSE_prime(y, y_hat, m):
    return (y - y_hat) / m


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return np.where(z > 0, 1.0, 0.0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


funcDict = {
    "sigmoid": sigmoid,
    "sigmoidPrime": sigmoidPrime,
    "relu": relu,
    "reluPrime": relu_prime,
    "MSE": MSE,
    "MSEPrime": MSE_prime
}


class Layer:

    def __init__(self, input_size, size, act_func="sigmoid"):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.rand(self.input_size, self.size)
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

    def __init__(self, input_size, output_size, cost_func = "MSE"):
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

    def backprop(self, x, y, m):
        yHat = self.forward(x)
        cost = funcDict[self.cost_func](y, yHat, m)
        delta = np.multiply(-funcDict[self.cost_func + "Prime"](y, yHat, m), funcDict[self.layers[-1].act_func+"Prime"](self.layers[-1].z))
        do = np.dot(self.layers[-1].x.T, delta)
        der = []
        db = []
        der.insert(0, do)
        db.insert(0, np.sum(delta, axis=0))
        for i in reversed(range(len(self.layers))):
            if self.layers[i] == self.layers[-1]:
                continue
            delta = np.dot(delta, self.layers[i + 1].weights.T) * funcDict[self.layers[i].act_func+"Prime"](self.layers[i].z)
            der.insert(0, np.dot(self.layers[i].x.T, delta))
            db.insert(0, np.sum(delta, axis=0))
        return der, db, cost

    def train(self, lr, x, y, m, batch_size=32, epochs=1):
        der, db, cost = self.backprop(x, y, m)
        costs = [cost]

        for itr in range(epochs):
            mini_batches_x, mini_batches_y = create_mini_batches(x, y, batch_size)
            np.random.shuffle(mini_batches_x)
            np.random.shuffle(mini_batches_y)
            for batch in range(len(mini_batches_x)):
                der, db, cost = self.backprop(mini_batches_x[batch], mini_batches_y[batch], m)

                for i in range(len(self.layers)):
                    descent = lr * der[i]
                    biasDescent = lr * db[i]
                    self.layers[i].weights -= descent
                    self.layers[i].bias -= biasDescent
            costs.append(cost)
        plt.plot(costs)
        plt.show()

    def test(self, x, y, iterations):
        accuracy = 0
        print(y)
        for j in range(iterations):
            index = int(np.random.rand(1) * len(x))
            test = x[index]
            a = self.forward(test)
            if a >= .5:
                b = 1
            else:
                b = 0

            print("Input: {}, Actual: {}, Desired: {}".format(test, [a], y[index]))
            if b == y[index]:
                accuracy += 1
        print("Accuracy: {}".format(accuracy / iterations))
