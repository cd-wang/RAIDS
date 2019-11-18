from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from config import TrainConfig




def create_comma_model_large_dropout(row,col,ch, load_weights=False):  #change## parameter values // deepxplore Dave_dropout
    model = Sequential()

    model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())

    model.add(Dense(500))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(.25))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    if load_weights:
        model.load_weights('./Model.h5')

    print('Model is created and compiled..')
    return model

