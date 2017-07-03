# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:26:31 2017

@author: cangye@geophyx.com
"""

import tensorflow as tf
import numpy as np
class FaceIdentify():
    """
    Define the trunk process
    """
    def __init__(self, data_len, vali_len, data_for_test=None):
        self.conv_shape = {
            "conv_layer1": {"w":[3, 3, 3, 8], "b":[8], "s":[1,1,1,1], "p":[1,1,1,1], "ps":[2]},
            "conv_layer2": {"w":[3, 3, 8, 16], "b":[16], "s":[1,1,1,1], "p":[1,1,1,1], "ps":[1]},
            "conv_layer3": {"w":[3, 3, 16, 16], "b":[8], "s":[1,1,1,1], "p":[1,1,1,1], "ps":[1]},
            "conv_layer4": {"w":[3, 3, 16, 16], "b":[1], "s":[1,1,1,1], "p":[1,1,1,1], "ps":[1]},

            "conv_layer5": {"w":[4, 3, 8], "b":[8], "s":2, "p":[4], "ps":[1]},
            "conv_layer6": {"w":[4, 3, 8], "b":[8], "s":2, "p":[4], "ps":[1]},
            "conv_layer7": {"w":[4, 3, 8], "b":[8], "s":2, "p":[4], "ps":[1]},
            "conv_layer8": {"w":[4, 3, 8], "b":[8], "s":2, "p":[4], "ps":[1]}
        }
        self.full_shape = {
            "full_layer1":[[14, 12], [12]],
            "full_layer2":[[12, 1], [1]]
        }
        self.conv_names = ["conv_layer1", "conv_layer2", "conv_layer3", "conv_layer4"]
        self.full_names = ["full_layer1", "full_layer2"]
        self.sess = tf.Session()
        self.layers_data = {}
        self.data_for_test = data_for_test
        self.global_train_count = 0
        self.data_len = data_len
        self.vali_len = vali_len

        self.def_struct()

    def def_struct(self):
        """
        Generate NN struct
        """
        with tf.variable_scope("input_layer"):
            self.layers_data["input_layer"] = tf.placeholder(tf.float32,
                                                             [None, self.data_len, 3])
            perior = "input_layer"
            self.vali = tf.placeholder(tf.float32, [None, self.vali_len])
        for c_l in self.conv_names:
            with tf.variable_scope(c_l):
                self.layers_data[c_l] = self.conv_relu(self.layers_data[perior],
                                                       self.conv_shape[c_l])
                perior = c_l
        if self.data_for_test != None:
            self.sess.run(tf.global_variables_initializer())
            print(self.sess.run(tf.shape(self.layers_data[perior]),
                                feed_dict={self.layers_data["input_layer"]: np.zeros([50, self.data_len, 3])}))
        with tf.name_scope("train"):
            self.for_valid = tf.reduce_mean(tf.square(self.layers_data[perior] - self.vali))
            self.train_step = tf.train.AdamOptimizer(0.05).minimize(self.for_valid)
            tf.summary.scalar('cross_entropy', self.for_valid)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("logdir", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
    def conv_relu(self, data, pars):
        """
        Define the conf layer
        """
        weight = tf.get_variable("conv_weight", pars['w'],
                                 initializer=tf.random_normal_initializer())
        biases = tf.get_variable("conv_bias", pars['b'],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(data, filters=weight, stride=pars["s"],
                            padding='VALID', name="conv_data")
        #self.variable_summaries(weight)
        return tf.nn.pool(tf.nn.sigmoid(conv + biases),
                          window_shape=pars["p"],
                          pooling_type='AVG',
                          strides=pars['ps'],
                          padding='VALID')
    def full_connect(self, data, weight_shape, bias_shape):
        """
        Define the full connected layer
        """
        weight = tf.get_variable("full_connect_weight", weight_shape,
                                 initializer=tf.random_normal_initializer())
        biases = tf.get_variable("full_connect_bias", bias_shape,
                                 initializer=tf.constant_initializer(0.0))
        return tf.nn.sigmoid(tf.matmul(data, weight)+biases)

    def train(self, data, vali):
        """
        Training Process
        """
        self.sess.run(self.train_step,
                        feed_dict={self.layers_data["input_layer"]:data,
                                    self.vali:vali})
        self.global_train_count+=1
        if(self.global_train_count%2==0):
            print("step:%d"%self.global_train_count,
                  self.sess.run(self.for_valid,
                  feed_dict={self.layers_data["input_layer"]:data,
                  self.vali:vali}))
            summary, _ = self.sess.run([self.merged, self.train_step],
                                        feed_dict={self.layers_data["input_layer"]:data,
                                        self.vali:vali})
            self.train_writer.add_summary(summary, self.global_train_count)
    def valid(self, data, data_len):
        """
        Valid process
        """
        """
        Generate NN struct
        """
        self.vali_layers_data={}
        with tf.variable_scope("input_layer"):
            self.vali_layers_data["vali_input_layer"] = tf.placeholder(tf.float32,
                                                             [None, data_len, 3])
            perior = "vali_input_layer"
        for c_l in self.conv_names:
            with tf.variable_scope(c_l, reuse=True):
                self.vali_layers_data[c_l] = self.conv_relu(self.vali_layers_data[perior],
                                                       self.conv_shape[c_l])
                perior = c_l
        with tf.name_scope("get_data",):
            self.vali_data = self.vali_layers_data[perior]
        self.sess.run(tf.global_variables_initializer())
        return self.sess.run(self.vali_data, feed_dict={self.vali_layers_data["vali_input_layer"]:data})

class GenData():
    """
    Generate Data class
    """
    def __init__(self, shape):
        self.shape = shape
    def get_data_by_shape(self, shape=0):
        """
        gen data function
        """
        if shape != 0:
            return np.random.random(shape)
        return np.random.random(self.shape)

import matplotlib.pyplot as plt
from plotfig import *
if __name__ == "__main__":
    trunk = FaceIdentify(300,1)

    #print(genator.get_shape())
    