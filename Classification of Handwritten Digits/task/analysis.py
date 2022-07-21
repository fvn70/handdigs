import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

cols = 28 * 28
rows = x_train.shape[0]
x_train = x_train.reshape(rows, cols)
y_class = np.unique(y_train)

print('Classes: ', y_class)
print("Features' shape:", x_train.shape)
print("Target' shape:", y_train.shape)
print(f"min: {x_train.min()}, max: {x_train.max()}")
