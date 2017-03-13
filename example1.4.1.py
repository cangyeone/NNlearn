# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:32:52 2017

@author: Cangye
"""
#预读取MNIST手写字库
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

import tensorflow as tf
#用正态分布随机数初始化变量，本例中作为权值
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    #正态分布
    return tf.Variable(initial)
#用常量方式初始化偏置
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    #常数分布
    return tf.Variable(initial)
#定义二维卷积过程    
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#定义池化层，简单来说就是选个最大的数，进一步降低自由参数个数。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

x=tf.placeholder(tf.float32,shape=[100,784])
y=tf.placeholder(tf.float32,shape=[100,10])
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
x_image=tf.reshape(x,[-1,28,28,1])
y_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
y_pool1=max_pool_2x2(y_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
y_conv2 = tf.nn.relu(conv2d(y_pool1, W_conv2) + b_conv2)
y_pool2 = max_pool_2x2(y_conv2)
y_fc_flat= tf.reshape(y_pool2,[-1,7*7*64])
W_fc1 = weight_variable([7*7*64,10])
b_fc1 = bias_variable([10])
y_fc1=tf.nn.relu(tf.matmul(y_fc_flat,W_fc1)+b_fc1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_fc1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    bx,by=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:bx,y:by})

import numpy as np
import matplotlib.pyplot as plt
#为了画图更好看，其实并没有
import matplotlib as mpl
mpl.style.use('seaborn-darkgrid')

val=W_conv1.value()
convVal=np.array(sess.run(val))
convVal=np.reshape(convVal,[5,5,32])

plt.imshow(convVal[:,:,6])
plt.show()