'''
Main file to run, this will produce a video 
using MatplotLib and display the results 
within a GUI
'''

import matplotlib.pyplot as plt 
import matplotlib.image  as mimg
import os
import cv2
from models import GestureAngleClassifier

#Matplotlib jargon to setup GUI for one frame
def display_image(path, pred):
    plt.clf()
    plt.imshow(mimg.imread(path))
    plt.text(0,0,pred[0] + ' Similarity: ' + str(pred[1]))
    plt.pause(0.1)
    plt.draw()

#Displays the videos of image
def display(input_folder):
    SIZE = 20
    FONT_SIZE = 22

    model = GestureAngleClassifier('train_angles.npy', 'train_labels.npy') #Instantiating a model for classification
    
    #Calibrating the Matplotlib GUI variables
    plt.rcParams["figure.figsize"] = (SIZE,SIZE)
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.show()

    #Display using frame-by-frame basis
    for path in os.listdir(input_folder):
        display_image('./similarity_output/openpose_result.jpg', model.get_predictions(input_folder + '/' + path))

display('./shot2')