import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

cols = 28 * 28
rows = x_train.shape[0]
x_train = x_train.reshape(rows, cols)
y_class = np.unique(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train[:6000], y_train[:6000], train_size=0.7, random_state=40)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Proportion of samples per class in train set:')
print(pd.Series(y_train).value_counts(normalize=True))
