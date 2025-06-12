import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def softmax(z):
    ez = np.exp(z)
    sm = ez/np.sum(ez)
    return sm
plt.close("all")
plt_softmax(softmax)

#softmax loss calculation and training
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

model = Sequential([
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    Dense(units = 4, activation = 'softmax')
])
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = tf.keras.optimizers.Adam(0.001))
model.fit(X_train, y_train, epochs = 10)

#since the output layer activation is softmax, we get probabilities as outputs
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))

#now optimizing the calculation using from_logits and linear activation for the last layer
preferred_model = Sequential(
    [
        Dense(units = 25, activation = 'relu'),
        Dense(units = 15, activation = 'relu'),
        Dense(units = 10, activation = 'linear')
    ]
)
preferred_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), optimizer = tf.keras.optimizers.Adam(0.001))
preferred_model.fit(
    X_train,
    y_train,
    epochs = 10
)

#since we used linear activation for the last layer, output is not probabilistic, we need to apply the softmax to the output
p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))

sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))

#printing the most preferred in each X_train
for i in range(5):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")