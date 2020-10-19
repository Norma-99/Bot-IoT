import tensorflow
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

#import joblib
import pickle

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


x_train, y_train = load_data('datasets/train_dataset.pickle')
x_test, y_test = load_data('datasets/validation_dataset.pickle')


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
model.summary()

# fit the model
# opt = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=5, verbose=2)

