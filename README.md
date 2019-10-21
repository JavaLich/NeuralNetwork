# A$'s NeuralNetwork Library
A simple to use library for beginners to Neural Nets

## How to make a neural net

Making a network object with mean square error cost function:

`nn = NN.NeuralNetwork(num_of_inputs, num_of_outputs, "MSE")`

Possible cost functions:

- `"MSE"` Mean Square Error
- `"binaryCrossEntropy"` Binary Cross Entropy


Constructing the layers of the neural network:

Makes a network of 20 in the input layer, 1000 in the hidden layer, and 10 in the output layer with relu and sigmoid activations

```
nn.layer(NN.Layer(20, 1000, "relu"))
nn.layer(NN.Layer(1000, 10, "sigmoid"))
```

IMPORTANT: Layers must have same inputs as their previous layer's output

Possible activations:

- `"sigmoid"` Sigmoid
- `"relu"` Rectified Linear Unit
- `"linear"` Linear

Training your neural network:

```
nn.train(learning_rate, x_dataset, y_dataset, batch_size, epochs)
```

- learning_rate: Learning Rate (e.g. .005)

- x_dataset: Numpy array of input training dataset

- y_dataset: Numpy array of output training dataset

- batch_size: Batch size (Defaults to 32)

- epochs: number of epochs to train

Using your neural network:

```
nn.forward(x)
```

- x: input numpy array

Testing your neural network:

```
nn.test(x, y, iterations)
```

- x: Numpy input array
- y: Numpy output array
- iterations: iterations to test net

## Convolutional Neural Nets

Creating a Conv Net and connecting it to a deep neural net

```
conv = NN.ConvLayer(num_filters, (filter_shape), "relu", 2)
nn = NN.NeuralNetwork(13*13*8, 10)
nn.layer(NN.Layer(13*13*8, 10, "linear"))
cnn = NN.ConvNN(conv, nn)
```

- num_filters: number of filters to use
- filter_shape: shape of the filters in format (x, y)

Train and test a Conv Net

```
cnn.backprop(train_images[:1000], train_labels[:1000])
cnn.test(test_images[:100], test_labels[:100])
```