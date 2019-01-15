
"""
Isolated dataloader module
"""
import sys
import pickle
import numpy as np
import os.path as op
import tensorflow as tf
from sklearn.model_selection import train_test_split
from global_envs import *



def load_data_deprecated(point_data_path):
    """
    Function: load_data 
    parameters: point data path.
    returns: train_dataset test_dataset
    """
    if not op.exists(point_data_path):
        print('Error: %s not exits'%point_data_path)
        sys.exit(0)

    with open(point_data_path, 'rb') as data_file_handle:
        data = pickle.load(data_file_handle)

    features = list()
    labels = list()
    for item in data: 
        features.append(item[0])
        labels.append([item[1]])

    features = np.array(features).squeeze()
    labels = np.array(labels)

    features_train, features_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            train_size=0.8,
            random_state=33)

    train_dataset = tf.data.Dataset.from_tensor_slices(
            {'features_train': features_train,
                'labels_train': labels_train })

    test_dataset = tf.data.Dataset.from_tensor_slices(
            {'features_test': features_test,
                'labels_test': labels_test })

    return train_dataset, test_dataset


def load_data(point_data_path):
    if not op.exists(point_data_path):
        print('Error: %s not exits'%point_data_path)
        sys.exit(0)

    with open(point_data_path, 'rb') as data_file_handle:
        data = pickle.load(data_file_handle)

    features = list()
    labels = list()
    for item in data: 
        features.append(item[0])
        labels.append([item[1]])

    features = np.array(features).squeeze()
    labels = np.array(labels)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=0.8, random_state=33)
    return train_features, test_features, train_labels, test_labels

def test():
    point_data_path = "../out/extract_pixel_value_func2.data.20190107_043546"
    train_features, test_features, train_labels, test_labels = load_data(point_data_path)
    print(train_features.shape)

if __name__ == '__main__':
    test()
