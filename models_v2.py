
"""
This is the sub-module to build all the training models.
"""
__author__ = "haihuam"

import os
import sys
import time
import pickle
import numpy as np
import os.path as op
import tensorflow as tf
from global_envs import *
from tqdm import tqdm
from data_loader import load_data
from utility import time_stamp, LQueue, debug_print

tf.logging.set_verbosity(tf.logging.INFO)

class Model1(object):

    def __init__(self,
            data_path,
            depth_path,
            params=None):

        self.info_collector = dict()
        self.model_name = self.__class__.__name__
        self.data_path = data_path
        self.depth_path = depth_path
        self.train_features, self.test_features, self.train_labels, self.test_labels = load_data(data_path, depth_path)
        self.pre_process()
        self.features = tf.placeholder(tf.float32, [None, 330])
        self.label = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = 0.001
        self.info_collector['Model Name'] = self.model_name
        self.info_collector['Learning Rate'] = self.learning_rate

    def pre_process(self):
        """
        Function: data preprocessing
        """
        self.train_features = np.mean(self.train_features, axis=2)
        self.test_features = np.mean(self.test_features, axis=2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, seed=0)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1, seed=0)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv1d(self, x, W):
        return tf.nn.conv1d(x, W, 2,'VALID')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def max_pool_2x1(self, x):
        # return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')
        return tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='SAME')

    def setup_cnn_model(self ):

        input_layer = tf.reshape(self.features, [-1, 11, 30, 1])
        
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d( 
                inputs=input_layer, 
                filters=32, 
                kernel_size=[2, 2], 
                padding="same",
                activation=tf.nn.relu)
        
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=[2, 2], 
                padding='same',
                strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[2, 4],
                padding="same",
                activation=tf.nn.relu)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(
                inputs=conv2, 
                pool_size=[2, 4], 
                padding='same',
                strides=1)
        
        # Convolutional Layer #2 and Pooling Layer #2
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[1, 2],
                padding="same",
                activation=tf.nn.relu)

        # Pooling Layer #2
        pool3 = tf.layers.max_pooling2d(
                inputs=conv3, 
                pool_size=[1, 4], 
                padding="same",
                strides=1)

        # Dense Layer
        pool3_flat = tf.reshape(pool3, [-1, 6 * 15 * 128])
        dense1 = tf.layers.dense(inputs=pool3_flat, units=2048)
        dropout = tf.nn.dropout(dense1, self.keep_prob)

        # Output Layer
        prediction = tf.layers.dense(inputs=dropout, units=1)

        # Calculate loss using mean squared error
        self.mse = tf.losses.mean_squared_error(prediction, self.label)
        tf.summary.scalar('mse', self.mse)

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse)

    def check_retune(self, checkpoint_dir):
        checkpoint_dir_update = str()
        if op.exists(checkpoint_dir):
            if 'retune' in checkpoint_dir:
                name_splitted = checkpoint_dir.split('_')
                iteration_num = int(name_splitted[-1])
                iteration_num += 1
                prefix = name_splitted[0:-1]
                prefix.append(str(iteration_num))
                checkpoint_dir_update = '_'.join(prefix)
            else:
                checkpoint_dir_update = checkpoint_dir + '_retune_1'
            return checkpoint_dir_update
        else:
            print("Error: %s not exists."%checkpoint_dir)
            sys.exit(0)

    def continue_on_train(self, checkpoint_dir, n_epoch, batch_size=64):
        """
        Base on previous training result, continue on training.
        """
        
        self.tf_summary()
        sess, saver = self.session_restore(checkpoint_dir) 
        result = str()
        checkpoint_dir_update = self.check_retune(checkpoint_dir)

        train_writer = tf.summary.FileWriter(checkpoint_dir_update + '/plot_train', sess.graph)
        test_writer = tf.summary.FileWriter(checkpoint_dir_update + '/plot_test')
        
        for epoch in tqdm(range(n_epoch)):
            start_time = time.time()
            batch_train_mse, batch_test_mse, n_batch = 0, 0, 0
            for x_train_a, y_train_a in self.minibatches(self.train_features, self.train_labels, batch_size, shuffle=True):
                sess.run(
                        self.train_step, 
                        feed_dict={
                            self.features: x_train_a, 
                            self.label: y_train_a, 
                            self.keep_prob: 0.7})

                fetch = {
                        "merged": self.merged,
                        "loss": self.mse
                        }


                _train_result = sess.run(fetch,
                        feed_dict={self.features: x_train_a,
                            self.label: y_train_a,
                            self.keep_prob: 1.0})

                _test_result = sess.run(fetch,
                        feed_dict={self.features: self.test_features,
                            self.label: self.test_labels,
                            self.keep_prob: 1.0})

                train_writer.add_summary(_train_result['merged'], epoch)
                train_writer.flush()
                test_writer.add_summary(_test_result['merged'], epoch)
                test_writer.flush()

                _train_mse = _train_result['loss']
                _test_mse = _test_result['loss']

                batch_train_mse += _train_mse
                batch_test_mse += _test_mse
                n_batch += 1
                train_mse = batch_train_mse / n_batch
                test_mse = batch_test_mse / n_batch



            print('Train-Scope-MSE:', train_mse )  
            print('Test-Scope-MSE:', test_mse)  
            QoR = ' '.join(['Train-Scope-MSE:', str(train_mse), 'Test-Scope-MSE:', str(test_mse)])


        self.info_collector['QoR'] = QoR
        self.info_collector['Epoch'] = epoch
        
        saver.save(sess, op.join(checkpoint_dir_update, self.model_name))
        self.record_result(checkpoint_dir_update)

    def tf_summary(self):
        tf.summary.scalar('mse', self.mse)
        self.merged = tf.summary.merge_all()

    def train(self, n_epoch=500, batch_size=64):
        """
        Training fuction.
        """
        self.tf_summary()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver( tf.trainable_variables())
        result = str()

        time_info = time_stamp()

        qlen = 1000
        train_mse_his = LQueue(qlen)
        test_mse_his = LQueue(qlen)

        checkpoint_dir = "_".join([self.model_name, time_info])

        train_writer = tf.summary.FileWriter(checkpoint_dir + '/plot_train', sess.graph)
        test_writer = tf.summary.FileWriter(checkpoint_dir + '/plot_test')


        for epoch in tqdm(range(n_epoch)):
            start_time = time.time()
            batch_train_mse, batch_test_mse, n_batch = 0, 0, 0
            for x_train_a, y_train_a in self.minibatches(self.train_features, self.train_labels, batch_size, shuffle=True):
                sess.run(
                        self.train_step, 
                        feed_dict={
                            self.features: x_train_a, 
                            self.label: y_train_a, 
                            self.keep_prob: 0.7})

                fetch = {
                        "merged": self.merged,
                        "loss": self.mse
                        }


                _train_result = sess.run(fetch,
                        feed_dict={self.features: x_train_a,
                            self.label: y_train_a,
                            self.keep_prob: 1.0})

                _test_result = sess.run(fetch,
                        feed_dict={self.features: self.test_features,
                            self.label: self.test_labels,
                            self.keep_prob: 1.0})

                train_writer.add_summary(_train_result['merged'], epoch)
                train_writer.flush()
                test_writer.add_summary(_test_result['merged'], epoch)
                test_writer.flush()

                _train_mse = _train_result['loss']
                _test_mse = _test_result['loss']

                batch_train_mse += _train_mse
                batch_test_mse += _test_mse
                n_batch += 1
                train_mse = batch_train_mse / n_batch
                test_mse = batch_test_mse / n_batch



            print('Train-Scope-MSE2:', train_mse )  
            print('Test-Scope-MSE2:', test_mse)  
            QoR = ' '.join(['Train-Scope-MSE2:', str(train_mse), 'Test-Scope-MSE2:', str(test_mse)])

            train_mse_his.Append(train_mse)
            test_mse_his.Append(test_mse)


            test_mse_is_descrease = test_mse_his.IsDecrese()
            train_mse_is_descrease = train_mse_his.IsDecrese()

            break_condition = not ( test_mse_is_descrease and train_mse_is_descrease)
            if break_condition:
                break

        self.info_collector['QoR'] = QoR
        self.info_collector['Epoch'] = n_epoch
            
        saver.save(sess, op.join('.', checkpoint_dir, self.model_name))
        self.record_result(checkpoint_dir)

    def session_restore(self, checkpoint_dir):
        """
        Restore a session
        """
        sess = tf.Session()
        saver = tf.train.Saver( tf.trainable_variables())
        sess.run(tf.global_variables_initializer())
        
        model_path = self.get_model_path(checkpoint_dir)
        saver.restore(sess, model_path)
        return sess, saver

    def record_result(self, checkpoint_dir):
        result = str(self.info_collector)
        self.dump_result(checkpoint_dir, result)

    def dump_result(self, checkpoint_dir, result):
        result_file_dir = op.join(checkpoint_dir, 'result')
        with open(result_file_dir, 'w+') as result_file_handle:
            result_file_handle.write(result)
            result_file_handle.write('\n')

    def get_model_path(self, checkpoint_dir):
        from glob import glob
        meta_file = glob(op.join(checkpoint_dir,'*.meta'))
        if meta_file:
            if len(meta_file) == 1:
                meta_name = meta_file[0].split('/')[-1]
                model_name = meta_name.split('.')[0]
                model_path = op.join(checkpoint_dir, model_name)
                return model_path
            else:
                print('Error: more models were saved in [%s]'%checkpoint_dir)
                sys.exit(0)
        else:
                print('Error: no model found in [%s]' %checkpoint_dir)
                sys.exit(0)

    def minibatches(self,
            inputs=None,
            targets=None,
            batch_size=None,
            shuffle=False):

        assert len(inputs) == len(targets)

        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]



class Model2(Model1):

    def setup_cnn_model(self):

        input_layer = tf.reshape(self.features, [-1, 330, 1])
        
        # conv1 layer 
        W_conv1 = self.weight_variable([2, 1, 32]) 
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv1d(input_layer, W_conv1) + b_conv1) 
        # print(h_conv1.get_shape().as_list())
        # sys.exit(0)

        # pooling
        h_pool1 = self.max_pool_2x1(h_conv1)   

        # MS Layer
        ## Branch 1
        ms_b1_W_conv1 = self.weight_variable([1, 32, 10])
        ms_b1_b_conv1 = self.bias_variable([10])
        ms_b1_h_conv1 = tf.nn.relu(self.conv1d(h_pool1, ms_b1_W_conv1) + ms_b1_b_conv1)
        ms_b1_W_conv2 = self.weight_variable([5, 10, 10])
        ms_b1_b_conv2 = self.bias_variable([10])
        ms_b1_h_conv2 = tf.nn.relu(self.conv1d(ms_b1_h_conv1, ms_b1_W_conv2) + ms_b1_b_conv2)
        # print(ms_b1_h_conv2.get_shape().as_list())

        ## Branch 2
        ms_b2_W_conv1 = self.weight_variable([1, 32, 10])
        ms_b2_b_conv1 = self.bias_variable([10])
        ms_b2_h_conv1 = tf.nn.relu(self.conv1d(h_pool1, ms_b2_W_conv1) + ms_b2_b_conv1)
        ms_b2_W_conv2 = self.weight_variable([3, 10, 10])
        ms_b2_b_conv2 = self.bias_variable([10])
        ms_b2_h_conv2 = tf.nn.relu(self.conv1d(ms_b2_h_conv1, ms_b2_W_conv2) + ms_b2_b_conv2)
        # print(ms_b2_h_conv2.get_shape().as_list())

        ## Branch 3
        ms_b3_W_conv1 = self.weight_variable([1, 32, 10])
        ms_b3_b_conv1 = self.bias_variable([10])
        ms_b3_h_conv1 = tf.nn.relu(self.conv1d(h_pool1, ms_b3_W_conv1) + ms_b3_b_conv1)
        # print(ms_b3_h_conv1.get_shape().as_list())

        ## Branch 4
        ms_b4_pool1 = self.max_pool_2x1(h_pool1)
        ms_b4_W_conv1 = self.weight_variable([1, 32, 10])
        ms_b4_b_conv1 = self.bias_variable([10])
        ms_b4_h_conv1 = tf.nn.relu(self.conv1d(ms_b4_pool1, ms_b4_W_conv1) + ms_b4_b_conv1)
        # print(ms_b4_h_conv1.get_shape().as_list())

        ## Branch 5
        ms_b5_W_conv1 = self.weight_variable([1, 32, 10])
        ms_b5_b_conv1 = self.bias_variable([10])
        ms_b5_h_conv1 = tf.nn.relu(self.conv1d(h_pool1, ms_b5_W_conv1) + ms_b5_b_conv1)
        ms_b5_W_conv2 = self.weight_variable([2, 10, 10])
        ms_b5_b_conv2 = self.bias_variable([10])
        ms_b5_h_conv2 = tf.nn.relu(self.conv1d(ms_b5_h_conv1, ms_b5_W_conv2) + ms_b5_b_conv2)
        # print(ms_b5_h_conv2.get_shape().as_list())

        # Concat layer
        concat_layer = tf.concat([ms_b1_h_conv2, ms_b2_h_conv2, ms_b3_h_conv1, ms_b4_h_conv1, ms_b5_h_conv1], 1)
        # print(concat_layer.get_shape().as_list())
        # sys.exit(0)
        concat_layer_flat = tf.reshape(concat_layer, [ -1, 144*10])
        # print(concat_layer_flat.get_shape().as_list())

        # fc1 layer 
        W_fc1 = self.weight_variable([144*10, 512])
        b_fc1 = self.bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(concat_layer_flat, W_fc1) + b_fc1)

        # fc1 dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # fc2 layer 
        W_fc2 = self.weight_variable([512, 1])
        b_fc2 = self.bias_variable([1])

        self.prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.mse = tf.losses.mean_squared_error(self.label, self.prediction)
        tf.summary.scalar('mse', self.mse)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse)

    def predict(self, checkpoint_dir):
        """
        Load the training checkpoint, and implement the prediction.
        """
        sess, saver = self.session_restore(checkpoint_dir)
        prediction_value = sess.run(self.prediction,
                    feed_dict={ 
                        self.features: self.test_features,
                        self.keep_prob: 1.0})  
        return prediction_value


if __name__ == '__main__':

    data_path = "../out/extract_pixel_value.data.20190217_143517"
    depth_path = "../out/filterd_depth.info.20190217_125210"

    # model = Model1(data_path, depth_path)
    model = Model2(data_path, depth_path)
    model.setup_cnn_model()
    model.train()
