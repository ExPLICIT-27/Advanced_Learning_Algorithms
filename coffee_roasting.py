import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X, Y = load_coffee_data()
print(X.shape, Y.shape)
plt_roast(X, Y)

# Before normalization Max, Min
print(f"Temperature Max, Min pre nomralization: {np.max(X[:,0])}, {np.min(X[:, 0])}")
print(f"Duration Max, Min pre normalization: {np.max(X[:, 1])}, {np.min(X[:, 1])}")

#normalizing the values
norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(X) #learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post nomralization: {np.max(Xn[:,0])}, {np.min(Xn[:, 0])}")
print(f"Duration Max, Min post normalization: {np.max(Xn[:, 1])}, {np.min(Xn[:, 1])}")

#Tile/Copy our data to increase the training set size and decrease the number of epochs(iterations)

Xt = np.tile(Xn, (1000,1))
Yt = np.tile(Y, (1000, 1))
print(Xt.shape, Yt.shape)

#building the neural network
tf.random.set_seed(1234) #applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape = (2,)),
        Dense(3, activation = "sigmoid", name = "layer1"),
        Dense(1, activation = "sigmoid", name = "layer2")
    ]
)
model.summary()
#
"""
number of params = sum of parameters in all layers
in layer 1 : 3 neurons, input size has 2 cols:
therefore 2*3 + 3 (2*3 for 2 w parameters per neuron, 1 b per neuron)
in layer 2 : 2*1 + 1
total : 13
"""
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

#the model.compile defines a loss function and optimization
#the model.fit runs the gradient descent and fits weights to the data
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
)
model.fit(
    Xt, Yt,
    epochs = 10
)
"""#epochs is the number of iterations
In the fit statement above, the number of epochs was set to 10. 
This specifies that the entire data set should be applied during training 10 times.
 During training, you see output describing the progress of training that looks like this:

Epoch 1/10
6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
The first line, Epoch 1/10, describes which epoch the model is currently running. 
For efficiency, the training data set is broken into 'batches'. 
The default size of a batch in Tensorflow is 32. There are 200000 examples in our expanded data set or 6250 batches. 
The notation on the 2nd line 6250/6250 [==== is describing which batch has been executed.
"""
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

#Using the trained network to make predictions
X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("Predictions : \n", predictions)

yhat = (predictions >= 0.5).astype(int)
print(f"decisions : \n{yhat}")

#determining what each unit/neuron in the first layer is tasked with
#from the graph we can conclude tht each neuron is tasked in finding a different bad region
plt_layer(X, Y.reshape(-1, ), W1, b1, norm_l)

"""
The function plot of the final layer is a bit more difficult to visualize. 
It's inputs are the output of the first layer. We know that the first layer uses sigmoids so 
their output range is between zero and one. We can create a 3-D plot that calculates the output for all 
possible combinations of the three inputs. This is shown below. Above, high output values correspond to 
'bad roast' area's. Below, the maximum output is in area's where the three inputs are small 
values corresponding to 'good roast' area's."""
plt_output_unit(W2,b2)
#final graph showing the entire network in action
netf= lambda x : model.predict(norm_l(x))
plt_network(X,Y,netf)