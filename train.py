
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
from utility import time_stamp, LQueue
# from models import Model1, Model2, Model3, Model4, Model5, Model6
from models import *
# from models_v2 import Model1
from models import Model1
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
    func2_point_data = "extract_pixel_value_func2.data.20190107_043546"
    data_load_func2 = data_load_func1
    train_x, test_x, train_y, test_y = data_load_func2(func2_point_data)
    # checkpoint_dir = "func2_checkpoint_dir_5000_0.001_20190107_044853"
    model = Model(train_x, test_x, train_y, test_y, 0.001)
    # sys.exit(0)
    model.setup_net()
    # model.train(ex_name='func2')
    model.continue_on_train(checkpoint_dir, 100)

def model_loader(Model):
    func2_point_data = "extract_pixel_value_func2.data.20190107_043546"
    # func2_point_data = "../legacy_data/averaged_points.info"
    data_load_func2 = data_load_func1
    train_x, test_x, train_y, test_y = data_load_func2(func2_point_data)
    model = Model(train_x, test_x, train_y, test_y, 0.001)
    # model.setup_net()
    model.setup_net()
    model.train(ex_name='func2', epoch_limit=8000)

def model_predict(Model, checkpoint_dir):
    func2_point_data = "extract_pixel_value_func2.data.20190107_043546"
    data_load_func2 = data_load_func1
    train_x, test_x, train_y, test_y = data_load_func2(func2_point_data)
    # checkpoint_dir = "func2_checkpoint_dir_5000_0.001_20190107_044853"
    # print(train_x.shape)
    model = Model(train_x, test_x, train_y, test_y, 0.0005)
    # sys.exit(0)
    model.setup_net()
    # model.train(ex_name='func2')
    prediction_value = model.test_predict(checkpoint_dir)
    print(prediction_value['losses'])
    print(prediction_value['prediction'].shape)
    print(test_y.shape)
    # print(test_y)
    index = 0
    for i in range(0, len(test_y)):
        print(prediction_value['prediction'][i], test_y[i], prediction_value['prediction'][i] - test_y[i])
    
    return prediction_value


def train_func():
    # model_loader(Model1)
    # model_loader(Model2)
    # model_loader(Model3)
    # model_loader(Model4)
    # model_loader(Model5)
    # model_loader(Model6)
    model_loader(Model10)
    # prediction_value = model_predict(Model1, 'Model1_func2_20190115_144146')
    # print(prediction_value)


if __name__ == "__main__":
    train_func()

