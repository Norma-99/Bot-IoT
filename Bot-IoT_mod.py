import tensorflow
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import StandardScaler

#import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
"""Recopilar dataset"""

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()



def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(18, activation='relu', input_shape=(18,)))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(14, activation='sigmoid'))
    return model

# build the model
model = classification_model()

# fit the model
opt = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=5, verbose=2)

print(model.summary())
