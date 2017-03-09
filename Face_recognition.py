# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:31:15 2017

@author: harry
"""

import numpy as np
import matplotlib.pyplot as plt
from util import getData

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# load data
X, Y = getData(balance_ones=False)
print ("Data loaded ....\n")
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split( X, Y, test_size = 0.1, random_state = 0)
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 48, 48).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 48,48).astype('float32')

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

def larger_model(conv,drop,dense):
	# create model
    model = Sequential()
    model.add(Convolution2D(conv, 3, 3, border_mode='valid', input_shape=(1, 48, 48 ), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(conv, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(conv, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(dense/2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
scoreboard = []
conv = [52,96,99,120,123,144,147]
drop = [0.15,0.2,0.25,0.3]
dense = [128,256,512]
bestscore = 0
settings = [0,0,0]
for D in drop:
    for C in conv :
        for D2 in dense :
            print("Settings Conv, drop, dense : ", C, D, D2 )
            model = larger_model(C,D,D2)
            # Fit the model
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=30, batch_size=200, verbose=0)
            # Final evaluation of the model
            scores = model.evaluate(X_test, Y_test, verbose=0)
            
            print("Baseline Success: %.2f%%" % (scores[1]*100))
            if ((scores[1]*100) > bestscore) :  
                bestscore = (scores[1]*100)
                settings = [ C, D, D2 ]
            scoreboard.append(scores)

print ("-----------------------------------")
print (bestscore)
print (settings)
