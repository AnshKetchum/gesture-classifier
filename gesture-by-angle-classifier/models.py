import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Sequential
from angle_utils import get_angles
from openpose_data_for_single_image import get_single_image_data
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from os import path
from PIL import Image

class GestureAngleClassifier:

    output_labels = ['Forehand', 'Backhand', 'Overhead']

    train_input = []
    train_labels = []
    output_labels = []
    saved_file_name = 'weights2.h5'
    SIZE = 23


    classifier = Sequential([
            Flatten(input_shape = (,SIZE)) ,
            Dense(128, activation='relu')   ,
            Dense(len(output_labels), activation='softmax')
        ])
    
    def get_image(self,path):
        print('Getting requested image at: ',path)
        return cv2.imread(path)

    def format_image(self, path, p):
        return np.array(cv2.imread(path + '/' + p,0))

    def get_predictions(self, image_path):

        pred = np.array([get_angles(get_single_image_data(image_path))])
        prediction = self.classifier.predict(pred)
        index = np.argmax(prediction[0])
        return [self.output_labels[index]  , prediction[0][index]]

    def train_model(self):
        self.classifier.compile(optimizer= 'adam', loss ='sparse_categorical_crossentropy')
        self.classifier.fit(self.train_input, self.train_labels, epochs = 100)        

    def load_model(self):

            #Remember to add 'not' here   
        if not path.exists(self.saved_file_name):
            print('File not found, Creating ' + self.saved_file_name + ' file')
            print(self.train_input.shape)
            self.train_model()
            self.classifier.save_weights(self.saved_file_name)

        else:
            print(self.saved_file_name, ' was found')
            self.classifier.load_weights(self.saved_file_name)

    def __init__(self, train_input_path, train_labels_path):

        self.train_input = np.load(train_input_path)
        self.train_labels = np.load(train_labels_path)
        self.output_labels = self.train_labels

        print('Data Loaded.')
        self.load_model()
        print('NN Ready to Classify')
