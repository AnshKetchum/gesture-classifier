'''
This class serves as a third class 
to create a dataset for the neural network
to train on.
'''
import numpy as np
import os 
from angle_utils import get_angles
from openpose_data_for_single_image import get_single_image_data
import cv2

#Creates the two files (train input / labels) that will be used
def create_train_data():

    #Using same location as feature-based classifier 
    input_dir = '../gesture-by-feature-classifier/Dataset'

    #Input training files that will be generated for the neural network
    output_fileAngles = 'train_angles.npy'
    output_fileLabels = 'train_labels.npy'

    #Arrays that will be converted to numpy format for training data.
    angle_list = []
    outputs = []

    #Filling the angle_list (training angles array) and the output labels with data
    i = 0
    for label in os.listdir(input_dir):
        for image in os.listdir(input_dir + '/' + label + '/'):
            path = input_dir + '/' + label + '/' + image
            outputs.append(i)
            angle_list.append(get_angles(get_single_image_data(path)))
        i += 1

    #Convert arrays into numpy format and save them into files.
    angle_list = np.array(angle_list)
    np.save(output_fileAngles,angle_list)

    outputs = np.array(outputs)
    np.save(output_fileLabels,outputs)

create_train_data()