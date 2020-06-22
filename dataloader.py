# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 01:57:46 2020

@author: Kartik
"""

import numpy as np
import glob
import os
import cv2
import shutil
from sklearn.utils import shuffle
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from sklearn.metrics import confusion_matrix as cm


#function to extract faces from images and same them in a different folder
def create_dataset():
    if os.path.isdir('train_faces'):
        shutil.rmtree('train_faces')
    os.mkdir('train_faces')
    if os.path.isdir('test_faces'):
        shutil.rmtree('test_faces')
    os.mkdir('test_faces')
    
    
    path = os.getcwd() + "\\AR_FaceDB"
    #print("path:",path)
    not_included, crop_dim = {'train':[], 'test':[]}, (32,32)
    folders = glob.glob(path + "\\*")
    
    count = 0
    for i in folders:
        print(i)
        name = i.split("_")[-1]
        print("split name:",name)
        
        if name == "train":
            print("in train")
            label = (i.split("\\")[-1]).split("_")[0]
            for j in glob.glob(i+"\\*"):
                print(j)
                img = plt.imread(j)
                try:
                    img = detect_face_haar_cascade(img)
                    img = cv2.resize(img, crop_dim, interpolation = cv2.INTER_AREA)
                    img = cv2. cvtColor(img, cv2.COLOR_BGR2RGB)
                                        
                    #x_train.append(img)
                    if label == 'specs':
                        #y_train.append(1)
                        file_name = 'faces' + str(count) + '_' + str(1) + '.bmp'
                        cv2.imwrite(os.getcwd()+'\\train_faces\\'+file_name, img)
                    else:
                        #y_train.append(0)
                        file_name = 'faces' + str(count) + '_' + str(0) + '.bmp'
                        cv2.imwrite(os.getcwd()+'\\train_faces\\'+file_name, img)
                except:
                    print("face_not_found")
                    not_included['train'].append(j)
                count+=1
        elif name == "test":
            print("in test")
            label = (i.split("\\")[-1]).split("_")[0]
            for j in glob.glob(i+"\\*"):
                print(j)
                img = plt.imread(j)
                try:
                    img = detect_face_haar_cascade(img)
                    img = cv2.resize(img, crop_dim, interpolation = cv2.INTER_AREA)
                    img = cv2. cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    #x_test.append(img)
                    if label == 'specs':
                        #y_test.append(1)
                        file_name = 'faces' + str(count) + '_' + str(1) + '.bmp'
                        cv2.imwrite(os.getcwd()+'\\test_faces\\'+file_name, img)
                    else:
                        #y_test.append(0)
                        file_name = 'faces' + str(count) + '_' + str(0) + '.bmp'
                        cv2.imwrite(os.getcwd()+'\\test_faces\\'+file_name, img)
                    
                except:
                    print("face_not_found")
                    not_included['test'].append(j)
                count+=1
        else:
            print("Not doing anything about 'data' folder..")
    return not_included
