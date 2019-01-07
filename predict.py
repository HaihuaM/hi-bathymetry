
"""
This is the sub-module to load the train model to implement prediction.
"""
__author__ = "haihuam"

import os
import sys
import numpy as np
import os.path as op
import tensorflow as tf
from tqdm import tqdm
from utility import time_stamp
from train import Model1, data_load_func1
import matplotlib.pyplot as plt


def predict_model1():
    func1_point_data = "func1_point.data.20190106_091336"
    checkpoint_dir = "checkpoint_dir_Model1"
    train_x, test_x, train_y, test_y = data_load_func1(func1_point_data)
    # print(train_x.shape)
    model1 = Model1(train_x, test_x, train_y, test_y)
    model1.setup_net()
    model1.predict(checkpoint_dir)


if __name__ == "__main__":
    predict_model1()

