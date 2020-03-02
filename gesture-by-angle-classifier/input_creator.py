import numpy as np
import os 
from angle_utils import get_angles
import cv2

def infer_image(image):
    return 

#Creates the two files (train input / labels) that will be used
def create_train_data():
    input_dir = '../gesture-by-feature-classifier/Dataset'

    output_fileAngles = 'train_angles.npy'
    output_fileLabels = 'train_labels.npy'

    angle_list = []
    outputs = []

    for label in os.listdir(input_dir):
        for image in os.listdir(input_dir + '/' + label + '/'):
            outputs.append(label)
            angle_list.append(get_angles(infer_image(cv2.imread(image) ) ) )

    angle_list = np.array(angle_list)
    np.save(output_fileAngles,angle_list)

    outputs = np.array(outputs)
    np.save(output_fileLabels,outputs)

create_train_data()