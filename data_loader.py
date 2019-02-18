
"""
Isolated dataloader module
"""
import sys
import pickle
import numpy as np
import os.path as op
import tensorflow as tf
from utility import debug_print
from sklearn.model_selection import train_test_split
from global_envs import *


def load_data(data_path, depth_path):
    """
    Function: load data, depth data then split them into train set and validation set
    Parameters: data path, filtered depth data path
    Returns: train_features, test_features, train_labels, test_labels
    """

    with open(data_path, 'rb') as data_file_handle:
        data = pickle.load(data_file_handle)
        features = np.array(data)

    with open(depth_path, 'rb') as depth_file_handle:
        depth = pickle.load(depth_file_handle)
        _labels = np.array(depth)

    features = np.array(features).squeeze()
    _labels = np.array(_labels)
    labels = np.expand_dims(_labels, axis=1)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=0.8, random_state=33)
    # debug_print(train_labels.shape)

    return train_features, test_features, train_labels, test_labels

def test():
    data_path = "../out/extract_pixel_value.data.20190217_143517"
    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"
    train_features, test_features, train_labels, test_labels = load_data(data_path, depth_path)

if __name__ == '__main__':
    test()
