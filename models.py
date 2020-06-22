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

def start_training_tf(x_train, y_train):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            ])
    model.compile(optimizer='SGD',
                  loss='binary_crossentropy', #bcoz here it is binary classification mean_squared_error
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=64, shuffle=True)
    return model                

def start_training_tf_multitask(x_train, y_train):
    x_train = np.array([i.flatten() for i in x_train]) #to flatten all the input images
    
    #extracting features here
    x_in = tf.keras.Input(shape=(3072,))
    x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x_in)
    x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    
    #training saperate ouputs for multitasks now
    y1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    y1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(y1)
    
    y2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    y2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(y2)
    
    y3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    y3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(y3)
    
    y4 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    y4 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(y4)
    
    y5 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    y5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(y5)
    
    
    y1_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(y1)
    y2_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(y1)
    y3_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(y3)
    y4_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(y4)
    y5_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(y5)
    
    model = tf.keras.Model(inputs = x_in, outputs = [y1_out , y2_out, y3_out, y4_out, y5_out])
    
    model.compile(optimizer='SGD',
                  loss='binary_crossentropy', #bcoz here it is binary classification mean_squared_error
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=10, batch_size=64, shuffle=True)
    return model
