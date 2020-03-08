import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from os import path
from PIL import Image

class PureNNGestureClassifier:

    output_labels = ['Forehand', 'Backhand']

    train_input = []
    train_labels = []
    saved_file_name = 'weights.h5'
    SIZE = 224
    noTrain = False


    classifier = Sequential([
            Flatten(input_shape = (SIZE, SIZE,3)) ,
            Dense(128, activation='relu')   ,
            Dense(len(output_labels), activation='softmax')
        ])

    def output_image(self,image,out_dir):
        print('Creating a new image at: ', out_dir)
        imageio.imwrite(out_dir, image)
    
    def get_image(self,path):
        print('Getting requested image at: ',path)
        return cv2.imread(path)

    def format_image(self, path, p):
        return np.array(cv2.imread(path + '/' + p,0))

    #Creates the dataset, not fully implemented YET
    def load_data(self,path,output):
        count = 1
        for p in os.listdir(path):  
            if count > 1000:
                break
            self.train_input.append(self.get_image(path + '/' + p))
            self.train_labels.append(output)
            count += 1


    def get_predictions(self, image_to_classify):
        pred = np.array([image_to_classify])
        prediction = self.classifier.predict(pred)
        index = np.argmax(prediction[0])
        return [self.output_labels[index]  , prediction[0][index]]
    
    def get_accuracy(self, test_images, test_labels):
        test_loss, test_acc = self.classifier.evaluate(test_images, test_labels, verbose=2)
        print('Test Accuracy: ', test_acc)

    def train_model(self):
        self.classifier.compile(optimizer= 'adam', loss ='sparse_categorical_crossentropy')
        self.classifier.fit(self.train_input, self.train_labels, epochs = 100)        

    def load_model(self):

            #Remember to add 'not' here   
        if not path.exists(self.saved_file_name):
            print('File not found, Creating weights.h5 file')
            print(self.train_input.shape)
            self.train_model()
            self.classifier.save_weights(self.saved_file_name)
            self.noTrain = True

        else:
            print(self.saved_file_name, ' was found')
            self.classifier.load_weights(self.saved_file_name)

    def __init__(self, path_to_data, outputs):

        for i in range (len(path_to_data)):
            self.load_data(path=path_to_data[i], output=outputs[i])
        print('Data Loaded.')

        self.train_input = np.array(self.train_input)
        self.train_labels = np.array(self.train_labels)
        self.load_model()

        print('NN Ready to Classify')
