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

from dataloader import *
from models import *
from detect_face import *

#main
not_included = create_dataset() #Run this only to generate folder of train and test images
x_train, y_train, x_test, y_test = import_dataset()

tf_model = start_training_tf(x_train, y_train)
#tf_model_multi = start_training_tf_multitask(x_train, y_train_multi) #I need to build this y_train_multi on my own..........

print("Model error and accuracy on test set:", tf_model.evaluate(x_test, y_test))

cs_train, cn_train, cs_test, cn_test = count_classes(y_train, y_test)

[[tp, tn], [fp, fn]] = cm(y_test, tf_model.predict_classes(x_test))
print("1.Confusion matrix:")
print("",tp,"|",fp,"\n",fn,"|",tn)
print("2.Precision:", cal_precision(tp, fp))
print("3. Recall:", cal_recall(tp, fn))