'''
Utility class to manage dataset

'''

import cv2
import imutils
import os
from PIL import Image
import imageio
import numpy as np


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