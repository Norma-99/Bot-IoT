import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import joblib

def load_data(path):
    with open(path, 'rb') as f:
        return joblib.load(f)


x_train, y_train = load_data('datasets/train_dataset.pickle')
x_test, y_test = load_data('datasets/val_dataset.pickle')


def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(18, activation='relu', input_shape=(18,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(14, activation='sigmoid'))
    return model

# build the model
model = classification_model()

# fit the model
opt = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=5, verbose=2)

