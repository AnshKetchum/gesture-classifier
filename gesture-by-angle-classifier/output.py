import matplotlib.pyplot as plt 
import matplotlib.image  as mimg
import os
import cv2
from models import GestureAngleClassifier

def display_image(path, pred):
    plt.imshow(mimg.imread(path))
    plt.text(0,0,'Classification: ', pred[0], ' and similarity: ', pred[1])
    plt.pause(0.5)
    plt.show()


def display(input_folder):
    model = GestureAngleClassifier('train_angles.npy', 'train_labels.npy')

    for path in os.listdir(input_folder):
        display_image('./similarity_output/openpose_result.jpg', model.get_predictions(input_folder + '/' + path))

display('./shot2')