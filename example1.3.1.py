# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:18:49 2017

@author: LLL
"""

N=200

import tensorflow as tf
import numpy as np


def Update(data2D):
    data4D=tf.reshape(data2D,[1,N,N,1])
    updateKernel2D=tf.constant([[0.0, 1.0, 0.0],
                           [1.0, -2., 1.0],
                           [0.0, 1.0, 0.0]],dtype=tf.float32)
    updateKernel4D=tf.reshape(updateKernel2D,[3,3,1,1])
    newData4D=tf.nn.conv2d(data4D,updateKernel4D,[1,1,1,1],padding='SAME')
    return tf.reshape(newData4D,[N,N])


#sess = tf.InteractiveSession()
initT=tf.zeros([N,N])
initS=tf.zeros([N,N])

for _ in range(10):
    ix,iy=np.random.randint(0,N,2)
    initS[ix,iy]=np.random.uniform()

Tn1=tf.Variable(initS,dtype=tf.float32)
Tn2=tf.Variable(initT,dtype=tf.float32)

eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())


Tn1_t=Tn1+eps*Tn2
Tn2_t=Tn1+eps*(Tn1-damping*Tn2)

step=tf.group(Tn1.assign(Tn1_t),Tn2.assign(Tn2_t))

sess=tf.Session()
init=tf.global_variables_initializer()

sess.run(init)

for _ in range(1000):
    sess.run(step,feed_dict={eps: 0.03, damping: 0.04})
    
import matplotlib.pyplot as plt

plt.imshow(sess.run(Tn1.value))
plt.show()
    
    
    
    
    
    
    
    
    
    
    