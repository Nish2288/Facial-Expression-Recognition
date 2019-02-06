import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as k

class LeNet:
    @staticmethod
    def build(width, height, channels, classes):
        inputshape = (height, width, channels)
        if k.image_data_format == 'channel_first':
            inputshape = (channels, height, width)
            
        model = Sequential()
        model.add(Conv2D(20, (5, 5), input_shape = inputshape, padding = 'same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size =  (2, 2), strides = (2, 2)))
        model.add(Conv2D(50, (5, 5), padding = 'same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2 ,2)))
        model.add(Flatten())
        model.add(Dense(500, activation = 'relu'))
        model.add(Dense(classes, activation = 'softmax'))

        return model

