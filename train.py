
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
from utility import time_stamp, LQueue
from models import Model1, Model2, Model3, Model4, Model5
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
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=33)
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
    data_load_func2 = data_load_func1
    train_x, test_x, train_y, test_y = data_load_func2(func2_point_data)
    model = Model(train_x, test_x, train_y, test_y, 0.001)
    model.setup_net()
    model.train(ex_name='func2')



def train_func():
    model_loader(Model1)
    model_loader(Model2)
    model_loader(Model3)
    model_loader(Model4)
    model_loader(Model5)


if __name__ == "__main__":
    train_func()

