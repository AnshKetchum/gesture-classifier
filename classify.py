import os
from models import PureNNGestureClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
import numpy as np
import cv2


input_directory = './Dataset'
inputs = []
outputs = []

def load_data():
    print('----------------------')
    count = 0
    for path in os.listdir(input_directory):
        inputs.append(input_directory + '/' + path)
        outputs.append(count)
        print('Label: ', path, ' and index: ', count)
        count += 1
    print('----------------------')


def read_image(path,SIZE):
    img = cv2.imread(path)
    resized = cv2.resize(img, (SIZE,SIZE), interpolation = cv2.INTER_AREA)
    return resized

def classify_shot_percentages(frame_directory, THRESHOLD_FOR_CLASSIFICATION, SIZE):
    classifier = PureNNGestureClassifier(inputs,outputs)
    i = 1
    frame_range = []
    similarity = []
    actions = []
    current_range = [0,0]
    current_similarity = 1.0
    current_action = 'none'

    for path in os.listdir(frame_directory):
        pred = classifier.get_predictions(read_image(frame_directory + '/' + path, SIZE))
        if pred[1] >= THRESHOLD_FOR_CLASSIFICATION and (current_action == 'none' or current_action == pred[0]):
            current_action = pred[0]
            if current_range[0] == 0 and current_range[1] == 0:
                current_range[0] = current_range[1] = i
                current_similarity = pred[1]
            else:
                current_range[1] = i
                current_similarity *= pred[1]
        else:
            if not (current_range[0] == 0 and current_range[1] == 0):
                frame_range.append(current_range)
                similarity.append(current_similarity)
                actions.append(current_action)
                current_range = [0,0]
                current_similarity = 1
            else:
                current_range = [0,0]
                current_similarity = 1
        i += 1

    if not (current_range[0] == 0 and current_range[1] == 0):
        frame_range.append(current_range)
        similarity.append(current_similarity)
        actions.append(current_action)

    print()
    print('Results')
    print('------------------------')
    print(frame_range)
    print(similarity)
    print(actions)
    print('------------------------')

#Only run this if you need to retrain
load_data() 

classify_shot_percentages('./shot2', 0.5,224)
#classifier = PureNNGestureClassifier([input_directory + '/Forhand'])