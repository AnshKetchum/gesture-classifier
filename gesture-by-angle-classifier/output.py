import matplotlib.pyplot as plt 
import matplotlib.image  as mimg
import os
import cv2
from models import GestureAngleClassifier

def display_image(path, pred):
    plt.imshow(mimg.imread(path))
    plt.pause(0.5)
    plt.show()


def display(input_folder):
    model = GestureAngleClassifier('train_angles.npy', 'train_labels.npy')

    for path in os.listdir(input_folder):
        pred = model.get_predictions(input_folder + '/' + path)
        pred = pred[0]

        display_image('./similarity_output/openpose_result.jpg', pred)

display('./shot2')