import NN
import numpy as np

x = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
x = np.divide(x, np.max(x))
y = np.array(([0], [1], [1], [0]))

nn = NN.NeuralNetwork(2, 1)
nn.layer(NN.Layer(2, 3))
nn.layer(NN.Layer(3, 1))
nn.train(.5, x, y, len(y), 50000)
nn.test(x, y, 10000)
