import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from helper import load_data
import keras
from sklearn.preprocessing import LabelBinarizer
from lenet import LeNet
from keras.preprocessing.image import ImageDataGenerator

faces, emotions = load_data()
#faces.shape
#emotions.shape

faces = faces/255.0

trainX, testX, trainY, testY = train_test_split(faces, emotions, test_size = 0.2, random_state = 20)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

data_gen = ImageDataGenerator(rotation_range=30, height_shift_range=0.2, width_shift_range=0.2, shear_range=0.2, horizontal_flip=True, zoom_range=0.3)

model = LeNet.build(width=64, height=64, channels=1, classes=7)
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics = ['accuracy'])

H = model.fit_generator(data_gen.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch = 28709//32, epochs=30, verbose=1)
model.save('cnn.hdf5')