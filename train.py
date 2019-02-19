
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
# from models import Model1, Model2, Model3, Model4, Model5, Model6
# from models import *
from models_v2 import *
# from models import Model1
import matplotlib.pyplot as plt


global OUTPUT_PATH

def data_load_func1(func1_point_data):
    """
    Function: load_func1_data 
    parameters: pixel data generate from func1.
    returns:
    """
    data_file = op.join(OUTPUT_PATH, func1_point_data)
    with open(data_file, 'rb') as data_file_handle:
        data = pickle.load(data_file_handle)

    features = list()
    labels = list()
    for item in data: 
        features.append(item[0])
        labels.append([item[1]])

    x = np.array(features).squeeze()
    y = np.array(labels)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.99, random_state=33)
    return train_x, test_x, train_y, test_y

def data_load_func3(data_file):
    """
    Function: load_func1_data 
    parameters: pixel data generate from func1.
    returns:
    """
    # data_file = op.join(OUTPUT_PATH, func1_point_data)
    with open(data_file, 'rb') as data_file_handle:
        data = pickle.load(data_file_handle)

    features = list()
    labels = list()

    for point in data:
        features.append(data[point][0])
        labels.append(data[point][1])

    x = np.array(features)
    y = np.array(labels)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=33)

    ss_x = preprocessing.StandardScaler()
    train_x_ = ss_x.fit_transform(train_x)
    test_x_ = ss_x.transform(test_x)

    ss_y = preprocessing.StandardScaler()
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    
    return train_x, test_x, train_y, test_y

def model_retune(Model, checkpoint_dir):

    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"
    model = Model(data_path, depth_path)
    model.setup_cnn_model()
    model.continue_on_train(checkpoint_dir, 500)

def model_loader(Model):

    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"
    model = Model(data_path, depth_path)
    model.setup_cnn_model()
    model.train()

def model_predict(Model, checkpoint_dir):
    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"
    
    model = Model(data_path, depth_path)
    model.setup_cnn_model()
    # model.train(ex_name='func2')
    prediction_value = model.predict(checkpoint_dir)
    prediction_value = np.squeeze(prediction_value)
    labels = np.squeeze(model.test_labels)

    predict_list = list()
    labels_list = list()
    # debug_print(prediction_value)
    index = 0
    for i in range(0, len(model.test_labels)):
        prediction_value[i])
        int(model.test_labels[i])
    
    
    return prediction_value


def train_func():
    # model_loader(Model1)
    # model_loader(Model2)
    # model_loader(Model3)
    # model_loader(Model4)
    # model_loader(Model5)
    # model_loader(Model6)
    # model_loader(Model2)
    # model_predict(Model2, 'Model2_20190218_205754')
    # model_retune(Model2, 'Model2_20190218_205754')
    model_retune(Model2, 'Model2_20190218_205754_retune_2')
    # prediction_value = model_predict(Model1, 'Model1_func2_20190115_144146')
    # print(prediction_value)


if __name__ == "__main__":
    train_func()

