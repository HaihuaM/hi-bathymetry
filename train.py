
"""
This is the sub-module to kick off the training models.
"""
__author__ = "haihuam"

import os
import sys
import pickle
import numpy as np
import os.path as op
import tensorflow as tf
from global_envs import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utility import time_stamp, LQueue, debug_print
from coordinate_transform import obj_dump
# from models import Model1, Model2, Model3, Model4, Model5, Model6
# from models import *
from models_v2 import *
# from models import Model1
import matplotlib.pyplot as plt


global OUTPUT_PATH


def model_retune(Model, checkpoint_dir):

    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"
    model = Model(data_path, depth_path)
    model.setup_cnn_model()
    model.continue_on_train(checkpoint_dir, 1000)

def model_loader(Model):

    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"
    model = Model(data_path, depth_path)
    model.setup_cnn_model()
    model.train(2000)

def model_predict(Model, checkpoint_dir):
    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"
    
    model = Model(data_path, depth_path)
    debug_print(model.train_features.shape)

    model.setup_cnn_model()
    
    prediction_value = model.predict(checkpoint_dir)
    prediction_value = np.squeeze(prediction_value)
    
    labels = np.squeeze(model.test_labels)

    predict_list = list()
    labels_list = list()

    index = 0
    for i in range(0, len(labels)):
        
        labels_list.append(float(labels[i]))
        predict_list.append(prediction_value[i])
        
    predict_file = "predict.info"
    labels_file = "label.info"
    obj_dump(predict_list, file_name=predict_file)
    obj_dump(labels_list, file_name=labels_file)

def train_func():
    # model_loader(Model1)
    # model_loader(Model2)
    # model_loader(Model3)
    # model_loader(Model4)
    # model_loader(Model5)
    # model_loader(Model6)
    # model_loader(Model2)
    model_predict(Model2, 'Model2_20190219_143221_retune_9')
    # model_retune(Model2, 'Model2_20190218_205754')
    # prediction_value = model_predict(Model1, 'Model1_func2_20190115_144146')
    # print(prediction_value)


if __name__ == "__main__":
    train_func()

