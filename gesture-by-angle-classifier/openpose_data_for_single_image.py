#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:23:19 2020

@author: sanath (Not we, NOTE: We (Ansh, Stephen) DO NOT TAKE CREDIT FOR THE CREATION OF THIS FILE)
"""

#Shot similarity prediction using openpose API 

import sys
import cv2
import os
#import math
import argparse
import numpy as np
#import  matplotlib.pyplot as plt
sys.path.append('/home/sanathv/work/openpose/build/python')
#Change this path to your openpose installation directory 

from openpose import pyopenpose as op

# Flags
#this script is specifically designed to get openpose data from API and works on 400 x 400 resolution 
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='/home/sanathv/work/openpose/models/')
parser.add_argument('--image_path', type=str, default='/home/sanathv/work/openpose/examples/media/COCO_val2014_000000000241.jpg')
parser.add_argument('--net_resolution', type=str, default='400x400')
#dont change resolution , I considered this for further processing of image
parser.add_argument('--number_people_max', type=int, default=1)
args = parser.parse_args()

# Custom Params
params = dict()
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = True

#starting openpose wrapper with above parameters  
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

shot_name = {'Forehand':0 , 'Backhand':1 ,'Overhead':2}
get_shot_name = { 0 : 'Forehand' , 1:'Backhand' , 2:'Overhead'}
pose_map={0:"Nose" ,1:"Neck",2:"RShoulder",3:"RElbow", 4:"RWrist",5:"LShoulder"
          ,6:"LElbow",7:"LWrist",8:"MidHip",9:"RHip",10:"RKnee",11:"RAnkle"
          ,12:"LHip",13:"LKnee",14:"LAnkle",15:"REye",16:"LEye",17:"REar"
          ,18:"LEar",19:"LBigToe",20:"LSmallToe",21:"LHeel",22:"RBigToe" 
          ,23:"RSmallToe",24:"RHeel",25:"Background"}
def save_openpose_data(image,datum,imageName):
    out_dir="./similarity_output"
    if not os.path.exists(out_dir):
        os.mkdirs(out_dir)
    cv2.imwrite(str(os.path.join(out_dir,imageName))+".jpg",datum)


def transform_data(numpy_array,w,h):
    #Transforms data of shape (1,25,3) to (25,2)
    x=np.zeros((25,2))
    #print("current shape is " ,numpy_array.shape)
    if(numpy_array.shape==(1,25,3)):
        #print("data is ok")
        for _ in range(0,1):
            #we have only  only one person
            for each_row in range(0,25):
                x[each_row][0]=int((numpy_array[_][each_row][0]*400)/int(h))
                x[each_row][1]=int((numpy_array[_][each_row][1]*400)/int(w))
        #print(x)
        return x
    else :
        #This happens only when openpose fails to identify any person  in that image
        print("skipping this data & returning Null values")
        return x

def get_single_image_data(ImagePath,image_name="openpose_result"):
    datum = op.Datum()
    imageToProcess = cv2.imread(ImagePath) 
    datum.cvInputData = imageToProcess
    opWrapper.waitAndEmplace([datum])
    opWrapper.waitAndPop([datum])
    if type(datum.poseKeypoints) == np.ndarray and datum.poseKeypoints.shape==(1,25,3):
        data=transform_data(datum.poseKeypoints,imageToProcess.shape[0],imageToProcess.shape[1])
        save_openpose_data(imageToProcess,datum.cvOutputData,image_name)
    else :
        data=np.zeros((25,2))
        print("something wrong with file ",ImagePath)
        print("Replacing it with a  zero vector of shape ",data.shape) 
    return data    

openpose_data=get_single_image_data(str(args.image_path),"image_1")
print("Openpose data points are ")
print(openpose_data)