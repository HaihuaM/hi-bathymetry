
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
from sklearn.model_selection import train_test_split
from utility import time_stamp, LQueue

class Model1(object):

    def __init__(self, train_x, test_x, train_y, test_y, lr=0.001):
        self.info_collector = dict()
        self.model_name = self.__class__.__name__
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.xs = tf.placeholder(tf.float32, [None, 330])
        self.ys = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = lr
        self.info_collector['Model Name'] = self.model_name
        self.info_collector['Learning Rate'] = self.lr

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

        cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - prediction),
            reduction_indices=[1]))
        self.mse2 = tf.losses.mean_squared_error(self.ys,
                prediction)

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

    def train(self, ex_name=''):
        """
        Training fuction.
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver( tf.trainable_variables())
        

        qlen = 1000
        train_mse_his = LQueue(qlen)
        test_mse_his = LQueue(qlen)


        time_info = time_stamp()
        if ex_name:
            checkpoint_dir = "_".join([self.model_name, ex_name, time_info])
        else:
            checkpoint_dir = "_".join([self.model_name, time_info])

        epoch = 1
        while True:
            sess.run(self.train_step,
                    feed_dict={self.xs: self.train_x,
                        self.ys: self.train_y,
                        self.keep_prob: 0.7})

            train_mse = sess.run(self.mse2,
                    feed_dict={self.xs: self.train_x,
                        self.ys: self.train_y,
                        self.keep_prob: 1.0})

            test_mse = sess.run(self.mse2,
                    feed_dict={self.xs: self.test_x,
                        self.ys: self.test_y,
                        self.keep_prob: 1.0})

            train_mse_his.Append(train_mse)
            test_mse_his.Append(test_mse)

            test_mse_is_descrease = test_mse_his.IsDecrese()
            train_mse_is_descrease = train_mse_his.IsDecrese()

            print(epoch,'Train-Scope-MSE2:', train_mse)  
            print(epoch, 'Test-Scope-MSE2:', test_mse)  
            QoR = ' '.join(['Train-Scope-MSE2:',
                str(train_mse),
                'Test-Scope-MSE2:',
                str(test_mse)])

            epoch += 1
            break_condition = not ( test_mse_is_descrease and train_mse_is_descrease)
            if break_condition:
                break

        self.info_collector['QoR'] = QoR
        self.info_collector['Epoch'] = epoch

        saver.save(sess, op.join(checkpoint_dir, self.model_name))
        self.record_result(checkpoint_dir)

    def record_result(self, checkpoint_dir):
        result = str(self.info_collector)
        self.dump_result(checkpoint_dir, result)

    def dump_result(self, checkpoint_dir, result):
        result_file_dir = op.join(checkpoint_dir, 'result')
        with open(result_file_dir, 'w+') as result_file_handle:
            result_file_handle.write(result)

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

    def continue_on_train(self, checkpoint_dir, epoch):
        """
        Base on previous training result, continue on training.
        """
        
        sess, saver = self.session_restore(checkpoint_dir) 
        result = str()
        checkpoint_dir_update = self.check_retune(checkpoint_dir)
        
        for i in range(epoch):
                sess.run(self.train_step,
                        feed_dict={self.xs: self.train_x,
                            self.ys: self.train_y,
                            self.keep_prob: 0.7})
                train_mse = sess.run(self.mse2,
                        feed_dict={self.xs: self.train_x,
                            self.ys: self.train_y,
                            self.keep_prob: 1.0})
                test_mse = sess.run(self.mse2,
                        feed_dict={self.xs: self.test_x,
                            self.ys: self.test_y,
                            self.keep_prob: 1.0})

                print(i,'Train-Scope-MSE2:', train_mse)  
                print(i, 'Test-Scope-MSE2:', test_mse)  

                QoR = ' '.join(['Train-Scope-MSE2:',
                    str(train_mse),
                    'Test-Scope-MSE2:',
                    str(test_mse), '\n'])

        self.info_collector['QoR'] = QoR
        self.info_collector['Epoch'] = epoch
        
        saver.save(sess, op.join(checkpoint_dir_update, self.model_name))
        self.record_result(checkpoint_dir_update)

    def predict(self, checkpoint_dir):
        """
        Load the training checkpoint, and implement the predicition.
        """
        sess, saver = self.session_restore(checkpoint_dir)
        print('MSE2:',
                sess.run(self.mse2,
                    feed_dict={self.xs: self.test_x,
                        self.ys: self.test_y,
                        self.keep_prob: 1.0}))  

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

    def self_attention(self):
        xs_transpose = tf.transpose(self.xs)
        # attention_weight = tf.nn.softmax(tf.matmul(self.xs, xs_transpose))
        attention_weight_softmax = tf.nn.softmax(tf.matmul(xs_transpose, self.xs))
        attention_weight_variables = self.weight_variable([330, 330])
        attention_weight = tf.matmul(attention_weight_softmax, attention_weight_variables)
        return tf.matmul(self.xs, attention_weight)

    def setup_net(self):

        att_xs = self.self_attention()
        x_image = tf.reshape(att_xs, [-1, 11, 30, 1])

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




class Model3(Model1):

    def setup_net(self):

        x_image = tf.reshape(self.xs, [-1, 11, 30, 1])

        # conv1 layer 
        W_conv1 = self.weight_variable([2, 2, 1, 32]) 
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) 

        # pooling
        h_pool1 = self.max_pool_2x2(h_conv1)   
        # print(h_conv1.get_shape().as_list())
       
        # conv2 layer 
        W_conv2 = self.weight_variable([2, 2, 32, 64]) # patch 2x2
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
        # print(h_conv2.get_shape().as_list())

        # pooling
        h_pool2 = self.max_pool_2x2(h_conv2)   
        # print(h_pool2.get_shape().as_list())

        # conv3 layer 
        W_conv3 = self.weight_variable([2, 2, 64, 64]) # patch 2x2
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3) 
 
        # conv4 layer 
        W_conv4 = self.weight_variable([2, 2, 64, 128]) # patch 2x2
        b_conv4 = self.bias_variable([128])
        h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4) + b_conv4) 
        # print(h_conv4.get_shape().as_list())

        # fc1 layer 
        W_fc1 = self.weight_variable([3*8*128, 4096])
        b_fc1 = self.bias_variable([4096])
 
        h_conv4_flat = tf.reshape(h_conv4, [-1, 3*8*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # fc2 layer 
        W_fc2 = self.weight_variable([4096, 1])
        b_fc2 = self.bias_variable([1])

        prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - prediction), reduction_indices=[1]))
        self.mse2 = tf.losses.mean_squared_error(self.ys, prediction)

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)





class Model4(Model1):

    def setup_net(self):

        x_image = tf.reshape(self.xs, [-1, 11, 30, 1])

        # conv1 layer 
        W_conv1 = self.weight_variable([2, 2, 1, 32]) 
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) 

        # pooling
        h_pool1 = self.max_pool_2x2(h_conv1)   
        # print(h_conv1.get_shape().as_list())
       
        # conv2 layer 
        W_conv2 = self.weight_variable([2, 2, 32, 64]) # patch 2x2
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
        # print(h_conv2.get_shape().as_list())

        # pooling
        h_pool2 = self.max_pool_2x2(h_conv2)   
        # print(h_pool2.get_shape().as_list())

        # conv3 layer 
        W_conv3 = self.weight_variable([2, 2, 64, 64]) # patch 2x2
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3) 
 
        # conv4 layer 
        W_conv4 = self.weight_variable([2, 2, 64, 128]) # patch 2x2
        b_conv4 = self.bias_variable([128])
        h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4) + b_conv4) 

        # fc1 layer 
        W_fc1 = self.weight_variable([3*8*128, 4096])
        b_fc1 = self.bias_variable([4096])
 
        h_conv4_flat = tf.reshape(h_conv4, [-1, 3*8*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # fc2 layer 
        W_fc2 = self.weight_variable([4096, 1])
        b_fc2 = self.bias_variable([1])

        prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - prediction), reduction_indices=[1]))
        self.mse2 = tf.losses.mean_squared_error(self.ys, prediction)

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)






class Model5(Model4):

    def train(self, ex_name='', n_epoch=5000, batch_size=32):
        """
        Training fuction.
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver( tf.trainable_variables())
        result = str()

        time_info = time_stamp()

        if ex_name:
            checkpoint_dir = "_".join([self.model_name, ex_name, time_info])
        else:
            checkpoint_dir = "_".join([self.model_name, time_info])

        for epo in range(n_epoch):
            start_time = time.time()
            batch_train_mse, batch_test_mse, n_batch = 0, 0, 0
            for x_train_a, y_train_a in self.minibatches(self.train_x, self.train_y, batch_size, shuffle=True):
                sess.run(self.train_step, feed_dict={self.xs: self.train_x, self.ys: self.train_y, self.keep_prob: 0.7})
                _train_mse = sess.run(self.mse2, feed_dict={self.xs: self.train_x, self.ys: self.train_y, self.keep_prob: 1.0})
                _test_mse = sess.run(self.mse2, feed_dict={self.xs: self.test_x, self.ys: self.test_y, self.keep_prob: 1.0})
                batch_train_mse += _train_mse
                batch_test_mse += _test_mse
                n_batch += 1
                train_mse = batch_train_mse / n_batch
                test_mse = batch_test_mse / n_batch
                print('Train-Scope-MSE2:', train_mse )  
                print('Test-Scope-MSE2:', test_mse)  
                QoR = ' '.join(['Train-Scope-MSE2:', str(train_mse), 'Test-Scope-MSE2:', str(test_mse), '\n'])
            # time.sleep(1)


        self.info_collector['QoR'] = QoR
        self.info_collector['Epoch'] = n_epoch
            
        saver.save(sess, op.join('.', checkpoint_dir, self.model_name))
        self.record_result(checkpoint_dir)
        



