import numpy as np
import matplotlib.pyplot as plt


def MSE(y, y_hat):
    return .5 * np.sum((y - y_hat) ** 2)


def MSE_prime(y, y_hat):
    return y - y_hat


def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x))


def soft_max_prime(x):
    return -1 / x


def cross_entropy(y_hat):
    return -np.log(y_hat)


def binary_cross_entropy(y, y_hat):
    return -np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def binary_cross_entropy_prime(y, p):
    return (y / p) - ((1 - y) / (1 - p))


def soft_max_cross_entropy_prime(y, p):
    return p - y


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
        self.weights = np.random.rand(self.input_size, self.size) / self.input_size
        self.bias = np.zeros(self.size)
        self.z = 0
        self.x = np.zeros(self.weights.shape)
        self.act_func = act_func

    def derivative(self, x):
        return funcDict[self.act_func + "Prime"](self.z) * self.weights, funcDict[self.act_func + "Prime"](self.z) * x

    def evaluate(self, x):
        self.x = x
        self.z = np.dot(x, self.weights) + self.bias
        return funcDict[self.act_func](self.z)


class ConvLayer:

    def __init__(self, num_filters, filter_shape, act_func="relu", pool_size=1):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, filter_shape[0], filter_shape[1]) / (
                    filter_shape[0] * filter_shape[1])
        self.act_func = act_func
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.softmax_input_shape = 0
        self.softmax_input = 0

    def forward(self, image):
        output = np.zeros((image.shape[0] - (self.filter_shape[0] - 1), image.shape[1] - (self.filter_shape[1] - 1),
                           self.num_filters))
        for i in range(output.shape[1]):
            for j in range(output.shape[0]):
                region = image[i:(i + self.filter_shape[1]), j:(j + self.filter_shape[0])]
                output[i, j] = np.sum(region * self.filters, axis=(1, 2))
        self.softmax_input_shape = output.shape
        self.softmax_input = self.max_pool(output).flatten()
        return self.softmax_input

    def max_pool(self, image):
        output = np.zeros((image.shape[0] // self.pool_size, image.shape[1] // self.pool_size, self.num_filters))
        for u in range(self.num_filters):
            for i in range(0, image.shape[0], self.pool_size):
                for j in range(0, image.shape[1], self.pool_size):
                    pixel = np.max(image[i:(i + self.pool_size), j:(j + self.pool_size), u])
                    output[i // self.pool_size, j // self.pool_size, u] = pixel
        return output


def create_mini_batches(x, y, batch_size):
    mini_batches_x = []
    mini_batches_y = []
    for i in range(0, len(x), batch_size):
        mini_batches_x.append(x[i:i + batch_size])
        mini_batches_y.append(y[i:i + batch_size])

    return mini_batches_x, mini_batches_y


class ConvNN:

    def __init__(self, cLayer, NN):
        self.NN = NN
        self.cLayer = cLayer
        self.a = 0
        self.out = 0
        self.max = 0
        self.cost = 0
        self.totals = 0

    def forward(self, x, label):
        self.a = self.cLayer.forward(x) / 255
        self.out = self.NN.forward(self.a)
        self.totals = self.out
        self.max = soft_max(self.out)
        self.cost = cross_entropy(self.max[label])
        return self.max, self.cost

    def backprop(self, x, label, lr=.05):
        costs = []
        for i in range(len(x)):
            out, cost = self.forward(x[i], label[i])
            actual = np.zeros(self.NN.output_size)
            actual[label[i]] = 1
            delta = soft_max_cross_entropy_prime(actual, out)
            dldw = self.NN.layers[-1].x.reshape(self.NN.layers[-1].input_size, 1).dot(
                delta.reshape(self.NN.output_size, 1).T)
            dldb = delta
            der = []
            db = []
            der.insert(0, dldw)
            db.insert(0, dldb)

            for j in reversed(range(len(self.NN.layers))):
                if self.NN.layers[-1] == self.NN.layers[j]:
                    continue
                delta = np.dot(delta, self.NN.layers[j + 1].weights.T) * funcDict[self.NN.layers[j].act_func + "Prime"](
                    self.NN.layers[j].z)
                dldw = np.dot(self.NN.layers[j].x.reshape(self.NN.layers[j].x.shape[0], 1),
                              delta.reshape(delta.shape[0], 1).T)
                der.insert(0, dldw)
                db.insert(0, delta)

            for j in range(len(self.NN.layers)):
                self.NN.layers[j].weights -= lr * der[j]
                self.NN.layers[j].bias -= lr * db[j]

            costs.insert(-1, cost)
        plt.plot(costs)
        plt.show()

    def test(self, x, labels):
        accuracy = 0
        total = 0
        for i in range(len(x)):
            out, cost = self.forward(x[i], labels[i])
            prediction = np.argmax(out)
            if prediction == labels[i]:
                accuracy += 1
                total += 1
            if i % 101 == 0 and not i == 0:
                print(accuracy / 100)
                accuracy = 0
        print(total / len(x))


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
