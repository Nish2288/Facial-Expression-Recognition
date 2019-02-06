import pandas as pd
import cv2
import numpy as np


def load_data():
        dataset = pd.read_csv('dataset/fer2013.csv')
        pixels = dataset['pixels'].tolist()
        faces = []
        for pixel_list in pixels:
            face = [int(pixel) for pixel in pixel_list.split(' ')]
            face = np.asarray(face).reshape(48, 48)
            face = cv2.resize(face.astype('uint8'),(64, 64))
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(dataset['emotion']).as_matrix()
        return faces, emotions