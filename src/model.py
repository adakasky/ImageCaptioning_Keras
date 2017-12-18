from __future__ import division, print_function
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense


def AlexNet(input_shape=(224, 224, 3)):
    model = Sequential()
    
    model.add(ZeroPadding2D(padding=(1, 1), input_shape=input_shape))
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_2'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_3'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_4'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_5'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Flatten())
    
    model.add(Dense(4096, name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax', name='dense_3'))
    
    return model


def BiLSTM(max_features=2000, maxlen=100):
    model = Sequential()
    
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    
    return model
