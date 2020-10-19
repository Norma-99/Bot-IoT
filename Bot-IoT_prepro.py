#import tensorflow
#from tensorflow import keras

import keras

import pickle
import joblib

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.optimizers import SGD

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from sklearn.preprocessing import StandardScaler


#from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


train = pd.read_csv('../datasets/Best_Training.csv') #he cambiado el directorio
test = pd.read_csv('../datasets/Best_Testing.csv')
"""Recopilar dataset"""

traindf = train
testdf = test

#inicio preproceso training
train = train.to_numpy()
train = train[:, 1:20]
for i in range(0,2934801):
    #ip treatment
    train[i,1] = train[i,1].split(".")
    train[i,1] = train[i,1][-1]
    train[i,3] = train[i,3].split(".")
    train[i,3] = train[i,3][-1]

    #0xs treatment
    if "0x" in train[i,2]:
        train[i,2] = train[i,2].replace('0x','')
        train[i,2] = int(train[i,2], 16)
    if "0x" in train[i,4]:
        train[i,4] = train[i,4].replace('0x','')
        train[i,4] = int(train[i,4], 16)    



newCol1 = pd.Series(train[:,1], name='saddr')
newCol2 = pd.Series(train[:,2], name='sport')
newCol3 = pd.Series(train[:,3], name='daddr')
newCol4 = pd.Series(train[:,4], name='dport')

traindf.update(newCol1)
traindf.update(newCol2)
traindf.update(newCol3)
traindf.update(newCol4)

precedentdf = traindf.iloc[:, 1:16]#2934801 esta bien para el pre
targetdf = traindf.iloc[:, 16:19]#2934801 esta bien para el pre

precedentdf['saddr'] = precedentdf['saddr'].astype(int)
precedentdf['sport'] = precedentdf['sport'].astype(int)
precedentdf['daddr'] = precedentdf['daddr'].astype(int)
precedentdf['dport'] = precedentdf['dport'].astype(int)
precedentdf = pd.get_dummies(precedentdf)
#print(precedentdf.columns)
targetdf = pd.get_dummies(targetdf, dtype=np.float32)
#print(targetdf.columns)
maxs = precedentdf.max()
maxs = maxs.to_numpy()
precedent = precedentdf.to_numpy()

for j in range(0,precedent.shape[1]):
    precedent[:,j] = precedent[:,j]/maxs[j]

#precedent = StandardScaler().fit_transform(precedent)
target = targetdf.to_numpy()
#print(target.shape)
#fin preproceso training

#-----------------------------------------------------------

#inicio preproceso testing
test = test.to_numpy()
test = test[:, 1:20]
for i in range(0,733699):
    #ip treatment
    test[i,1] = test[i,1].split(".")
    test[i,1] = test[i,1][-1]
    test[i,3] = test[i,3].split(".")
    test[i,3] = test[i,3][-1]

    #0xs treatment
    if "0x" in test[i,2]:
        test[i,2] = test[i,2].replace('0x','')
        test[i,2] = int(test[i,2], 16)
    if "0x" in test[i,4]:
        test[i,4] = test[i,4].replace('0x','')
        test[i,4] = int(test[i,4], 16)

newCol1 = pd.Series(test[:,1], name='saddr')
newCol2 = pd.Series(test[:,2], name='sport')
newCol3 = pd.Series(test[:,3], name='daddr')
newCol4 = pd.Series(test[:,4], name='dport')

testdf.update(newCol1)
testdf.update(newCol2)
testdf.update(newCol3)
testdf.update(newCol4)

pre_testdf = testdf.iloc[:, 1:16]#2934801 esta bien para el pre
tar_testdf = testdf.iloc[:, 16:19]#2934801 esta bien para el pre

pre_testdf['saddr'] = pre_testdf['saddr'].astype(int)
pre_testdf['sport'] = pre_testdf['sport'].astype(int)
pre_testdf['daddr'] = pre_testdf['daddr'].astype(int)
pre_testdf['dport'] = pre_testdf['dport'].astype(int)
pre_testdf = pd.get_dummies(pre_testdf, dtype=np.float32)
#print(pre_testdf.columns)
tar_testdf = pd.get_dummies(tar_testdf, dtype=np.float32)

#add cols
test[:,1] = 0
auxCol = test[:,1]

tar_testdf.insert(6, 'subcategory_Data_Exfiltration', auxCol, True)
#print(tar_testdf.columns)

maxs = pre_testdf.max()
maxs = maxs.to_numpy()
pre_test = pre_testdf.to_numpy()

for j in range(0,pre_test.shape[1]):
    pre_test[:,j] = pre_test[:,j]/maxs[j]

tar_test = tar_testdf.to_numpy()
#print(pre_test.shape)
#print(tar_test.shape)
#print(tar_test[0])

#fin preproceso testing

validation_pair = pre_test, tar_test
training_pair = precedent, target
'''

with open('datasets/val_dataset.pickle.pickle', 'wb') as f:
    joblib.dump(validation_pair, f)

with open('datasets/train_dataset.pickle', 'wb') as f:
    joblib.dump(training_pair, f)
'''
