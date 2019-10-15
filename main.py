import NN
import numpy as np
import mnist
import matplotlib.pyplot as plt

train_images = mnist.train_images()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
train_labels = mnist.train_labels()
conv = NN.ConvLayer(8, (3, 3), "relu", 2)

nn = NN.NeuralNetwork(13*13*8, 10)
nn.layer(NN.Layer(13*13*8, 1000, "linear"))
nn.layer(NN.Layer(1000, 10, "linear"))
cnn = NN.ConvNN(conv, nn)
cnn.backprop(train_images, train_labels)
cnn.test(test_images, test_labels)

