import matplotlib.pyplot as plt 
import matplotlib.image  as mimg
import os
import cv2
from models import GestureAngleClassifier

def display_image(path, pred):
    plt.clf()
    plt.imshow(mimg.imread(path))
    plt.text(0,0,pred[0] + ' Similarity: ' + str(pred[1]))
    plt.pause(0.1)
    plt.draw()


def display(input_folder):
    model = GestureAngleClassifier('train_angles.npy', 'train_labels.npy')
    plt.show()

    for path in os.listdir(input_folder):
        display_image('./similarity_output/openpose_result.jpg', model.get_predictions(input_folder + '/' + path))

display('./shot2')