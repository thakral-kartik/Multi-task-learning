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


#face detection algos
def detect_face_mtcnn(img):
    #MTCNN
    detector = MTCNN()
    result = detector.detect_faces(img)
    bounding_box = result[0]['box']
    (x, y, w, h ) = bounding_box
    roi_color = img[y:y + h, x:x + w]
    return roi_color
    
def detect_face_haar_cascade(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32)
    )
    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_color = img[y:y + h, x:x + w]
        #roi_color = cv2.resize(roi_color, crop_dim, interpolation = cv2.INTER_AREA)
        return roi_color

def count_classes(y_train, y_test):
    '''
    There is a difference in count specs and count non-specs
    cs_train = 408
    cn_train = 269
    '''
    cs_train, cn_train, cs_test, cn_test = 0, 0, 0, 0
    for i in y_train:
        if i == 1:
            cs_train+=1
        elif i==0:
            cn_train+=1
        else:
            print("problem")
    for i in y_test:
        if i == 1:
            cs_test+=1
        elif i==0:
            cn_test+=1
        else:
            print("problem")
    return cs_train, cn_train, cs_test, cn_test

def confusion_matrix(m, x_test, y_test, thresh):
    '''
    reference: https://github.com/bhattbhavesh91/classification-metrics-python/blob/master/ml_a.ipynb
    '''
    pred = m.predict_proba(x_test)
        
    for i in range(len(pred)):
        if pred[i] >=thresh:
            pred[i]=1
        else:
            pred[i]=0
    pred = np.array(pred)
    
    tp = (sum((y_test==1) & (pred==1))).max()
    tn = (sum((y_test==0) & (pred==0))).max()
    fp = (sum((y_test==0) & (pred==1))).max()
    fn = (sum((y_test==1) & (pred==0))).max()
    
    
    return tp, tn, fp, fn

def cal_tpr(tp, fn):
    return tp / (tp + fn)

def cal_fpr(fp, tn):
    return fp / (fp + tn)

def cal_precision(tp, fp):
    return tp / (tp + fp)

def cal_recall(tp, fn):
    return tp / (tp + fn)

