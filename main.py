import NN
import numpy as np


data = np.loadtxt(open("USvideos.csv", "rb"), delimiter=",", skiprows=1)
data = data.T
category_id = data[0]
category_id -= np.mean(category_id)
category_id /= np.std(category_id)
likes = data[2]
likes -= np.mean(likes)
likes /= np.std(likes)
dislikes = data[3]
dislikes -= np.mean(dislikes)
dislikes /= np.std(dislikes)
comment_count = data[4]
comment_count -= np.mean(comment_count)
comment_count /= np.std(comment_count)
views = data[1]
views /= np.max(views)
x = np.array([category_id, likes, dislikes, comment_count])

x = x.T
x = np.delete(x, 30, 0)
x = np.vsplit(x, 2)
trainx = x[0]
testx = x[1]

print(trainx.shape)
views = views.T
views = np.delete(views, 30)
views = np.hsplit(views, 2)

trainy = views[0]
trainy = np.reshape(trainy, (trainy.shape[0], 1))
print(trainy.shape)
testy = views[1]
testy = np.reshape(testy, (testy.shape[0], 1))
x = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
print(x.shape)
x = np.divide(x, np.max(x))
y = np.array(([0], [1], [1], [0]))
nn = NN.NeuralNetwork(4, 1, "MSE")
nn.layer(NN.Layer(4, 16, "relu"))
nn.layer(NN.Layer(16, 1, "sigmoid"))
nn.train(.05, trainx, trainy, 29, 150)
nn.test(testx, testy, 1000)
print(np.mean(views[0]))

