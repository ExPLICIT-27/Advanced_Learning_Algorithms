import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#using the coffee data for the same
X, Y = load_coffee_data()
print(X.shape, Y.shape)
plt_roast(X, Y)

#normalizing the data
print(f"Temperature max min pre normalization: {np.max(X[:, 0])}, {np.min(X[:, 0])}")
print(f"Duration max min pre normalization: {np.max(X[:, 1])}, {np.min(X[:, 1])}")

norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(X)
Xn = norm_l(X)

print(f"Temperature max min post normalization: {np.max(Xn[:, 0])}, {np.min(Xn[:, 0])}")
print(f"Duration max min post normalization: {np.max(Xn[:, 1])}, {np.min(Xn[:, 1])}")

g = sigmoid

def my_dense(a_in, W, b):

    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)
    return p
X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
yhat = (predictions >= 0.5).astype(int)
print(f"Decisions = \n{yhat}")
netf= lambda x : my_predict(norm_l(x),W1_tmp, b1_tmp, W2_tmp, b2_tmp)
plt_network(X,Y,netf)