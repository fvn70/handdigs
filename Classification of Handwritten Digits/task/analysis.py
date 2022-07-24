import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(target_test, pred)

    print(f'Model: {model}\nAccuracy: {round(score, 3)}\n')
    return score


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

cols = 28 * 28
rows = x_train.shape[0]
x_train = x_train.reshape(rows, cols)
y_class = np.unique(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train[:6000], y_train[:6000], train_size=0.7, random_state=40)

models = (KNeighborsClassifier(),
          DecisionTreeClassifier(random_state=40),
          LogisticRegression(random_state=40, solver='liblinear'),
          RandomForestClassifier(random_state=40))

model_names = ['KNeighborsClassifier',
               'DecisionTreeClassifier',
               'LogisticRegression',
               'RandomForestClassifier']

best = (0, 0.)
for i in range(4):
    acc = fit_predict_eval(models[i], x_train, x_test, y_train, y_test)
    if acc > best[1]:
        best = i, acc

print(f'The answer to the question: {model_names[best[0]]} - {round(best[1], 3)}')
