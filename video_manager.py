import cv2
import os
from os import path



def create_output_dir():
    output_dir = '/Output'
    if not path.exists(output_dir):
        os.makedirs(output_dir)


def convert_to_frames(video_path):
    video  = cv2.VideoCapture()
    i = 1
    success = 1

    while success:
        success, image = video.read()
        cv2.imwrite("Frame-%d.jpg" % i, image)
        i += 1
