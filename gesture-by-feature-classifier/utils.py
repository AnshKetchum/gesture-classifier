import cv2
import imutils
import os
from PIL import Image
import imageio
import numpy as np

'''
class VideoHandler:
    vc = None
    
    def __init__(self, input_video_path):
        self.vc = cv2.VideoCapture(input_video_path)

    def rotate_90_degrees(frame):
        return imutils.rotate_bound(frame,90)

    def create_output_frame(frame,output_directory):
        cv2.imwrite(output_directory,frame)

    def video_to_images(output_directory,tag_name,image_extension):
        count = 1 
        success = 1
        
        if not(output_directory.endswith('/')):
            output_directory += '/'

        while success:
            success, frame = vc.read()
            frame = rotate_90_degrees(frame)
            
            out_dir = output_directory + tag_name + '-' + str(count) + image_extension
            create_output_frame(frame, out_dir)
            count += 1

        vc.release()
'''

class DatasetManager:

    count = 1

    #Takes in a filepath, returns the image
    def get_image(self,path):
        print('Getting requested image at: ',path)
        return Image.open(path)

    #Resizes the image and formats into easy-for-classification format
    def resize_and_format(self,image, SIZE):
        image = image.resize( (SIZE,SIZE) , Image.NEAREST)
        image = np.array(image)
        return image

    #Returns an image as output
    def output_image(self,image,out_dir):
        print('Creating a new image at: ', out_dir)
        imageio.imwrite(out_dir, image)

    #Creates a dataset folder
    def add_data(self,input_dir,output_dir, image_extension, SIZE):

        if not os.path.exists(output_dir):
            print('The directory: ' ,output_dir, ' did not exist. Creating it now')
            os.makedirs(output_dir)
        
        for path in os.listdir(input_dir):
            
            print('Checking path: ', input_dir + path)
            img = self.resize_and_format(self.get_image(input_dir + '/' +path), SIZE)

            out_dir = output_dir + 'Image-' + str(self.count) + image_extension
            self.output_image(img, out_dir)
            self.count += 1