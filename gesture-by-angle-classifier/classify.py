import os
from models import GestureAngleClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
import numpy as np
import cv2


inputs = 'train_angles.npy'
outputs = 'train_labels.npy'



model = GestureAngleClassifier(inputs, outputs)


#classifier = PureNNGestureClassifier([input_directory + '/Forhand'])