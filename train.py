
"""
This is the sub-module to train setup the training model.
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
from utility import time_stamp
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


class Model1(object):

    def __init__(self, train_x, test_x, train_y, test_y):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.xs = tf.placeholder(tf.float32, [None, 330])
        self.ys = tf.placeholder(tf.float32, [None, 1])
        #dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = 0.001

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    def setup_net(self):
        x_image = tf.reshape(self.xs, [-1, 11, 30, 1])

        # conv1 layer 
        W_conv1 = self.weight_variable([2, 2, 1, 32]) 
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) 
        # pooling
        h_pool1 = self.max_pool_2x2(h_conv1)   
       
        # conv2 layer 
        W_conv2 = self.weight_variable([2, 2, 32, 64]) # patch 2x2
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
 
        # fc1 layer 
        W_fc1 = self.weight_variable([6*15*64, 2048])
        b_fc1 = self.bias_variable([2048])
 
        h_pool2_flat = tf.reshape(h_conv2, [-1, 6*15*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # fc2 layer 
        W_fc2 = self.weight_variable([2048, 1])
        b_fc2 = self.bias_variable([1])

        prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - prediction), reduction_indices=[1]))
        self.mse2 = tf.losses.mean_squared_error(self.ys, prediction)

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)


    def train(self):
        """
        Training fuction.
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver( tf.trainable_variables())
        epoch = 5000
        time_info = time_stamp()
        checkpoint_dir = "_".join(["checkpoint_dir", str(epoch), str(self.lr), time_info])

        for i in range(epoch):
            sess.run(self.train_step, feed_dict={self.xs: self.train_x, self.ys: train_y, self.keep_prob: 0.7})
            # print(i,'Error:',sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1.0}))  
            print(i,'MSE2:',sess.run(self.mse2, feed_dict={self.xs: train_x, self.ys: train_y, self.keep_prob: 1.0}))  
 

        saver.save(sess, op.join('.', checkpoint_dir, 'Model_1'))

    def predict(self, checkpoint_dir):
        """
        Load the training checkpoint, and implement the predicition.
        """
        sess = tf.Session()
        saver = tf.train.Saver( tf.trainable_variables())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_dir)
        print('MSE2:',sess.run(self.mse2, feed_dict={self.xs: test_x, self.ys: test_y, self.keep_prob: 1.0}))  


if __name__ == "__main__":
    func1_point_data = "func1_point.data.20190106_091336"
    train_x, test_x, train_y, test_y = data_load_func1(func1_point_data)
    # print(train_x.shape)
    model1 = Model1(train_x, test_x, train_y, test_y)
    model1.setup_net()
    model1.train()

