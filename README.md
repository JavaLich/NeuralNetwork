# A$'s NeuralNetwork Library
A simple to use library for beginners to Neural Nets

Make a network with mean square error cost function

`nn = NN.NeuralNetwork(num_of_inputs, num_of_outputs, "MSE")`

Possible cost functions:

- `MSE`: Mean Square Error
- `binaryCrossEntropy`: Binary Cross Entropy

Construct the layers of the neural network

Makes a network of 20 in the input layer, 1000 in the hidden layer, and 10 in the output layer with relu and sigmoid activations

`nn.layer(NN.Layer(20, 1000, "relu"))
nn.layer(NN.Layer(1000, 10, "sigmoid"))`

