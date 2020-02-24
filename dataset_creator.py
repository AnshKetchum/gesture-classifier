
import os
from utils import DatasetManager

datasetHandler = DatasetManager()

input_directory = './UnsortedDataset'
output_directory = './Dataset'

#Creates the dataset folder, based on the unsorted dataset folder.
def create_dataset():
    for shot_type in os.listdir(input_directory):
        for shot in os.listdir(input_directory + '/' + shot_type):
            datasetHandler.add_data(input_dir=input_directory + '/' + shot_type + '/' + shot,output_dir=output_directory + '/' + shot_type + '/', image_extension='.jpg', SIZE= 224)

create_dataset()