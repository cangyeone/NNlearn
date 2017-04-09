# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:32:52 2017

@author: Cangye@hotmail.com
"""

import tensorflow as tf
import numpy as np

class GenDataXOR():
    def __init__(self,shape):
        self.shape=shape
    def func(self,dt):
        if(dt[0]+dt[1]<1):
            rt=[0.1]
        else:
            rt=[0.9]
        return rt
    def GenVali(self):
        self.vali=np.array(list(map(self.func,self.data)))
        return self.vali
    def GenData(self):
        self.data=np.random.random(self.shape)
        return self.data

x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])

W1=tf.Variable(tf.truncated_normal([2,1],stddev=0.1))
b1=tf.Variable(tf.constant(0.0,shape=[1]))
fc1=tf.nn.sigmoid(tf.matmul(x,W1)+b1)


ce=tf.reduce_mean(tf.square(fc1-y))
ts=tf.train.AdamOptimizer(1e-2).minimize(ce)

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
genData=GenDataXOR([50,2])
tsD=GenDataXOR([500,2])
ts1=tsD.GenData()
ts2=tsD.GenVali()
for i in range(6000):
    data=genData.GenData()
    vali=genData.GenVali()
    sess.run(ts,feed_dict={x:data,y:vali})
    if(i%1000==0):       
        print(sess.run(ce,feed_dict={x:ts1,y:ts2}))
#此之后为画图过程   
reW1=np.array(sess.run(W1.value()))
reb1=np.array(sess.run(b1.value()))
print(reW1)
print(reb1)
def sigmoid(rt):
    return 1/(1+np.exp(-rt))
def GenZ(X,Y):
    Z=np.zeros(np.shape(X))
    for ity in range(len(X)):
        for itx in range(len(X[0])):
            l1=np.matmul([X[ity,itx],Y[ity,itx]],reW1)+reb1
            l1f=sigmoid(l1)
            Z[ity,itx]=l1f[0]
    return Z
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.style.use('seaborn-darkgrid')
x=np.linspace(0,1,100)
y=np.linspace(0,1,100)
X,Y=np.meshgrid(x,y)
Z=GenZ(X,Y)
fig=plt.figure(1)
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,rstride=8,cstride=8, alpha=0.3)
ax.contour(X,Y,Z,zdir='z',offset=0, cmap=plt.cm.coolwarm)
def fmap(mm):
    if(mm>0.5):
        return 'm'
    else:
        return 'r'
st1=np.transpose(ts1)
plt.figure(2)
plt.scatter(st1[0],st1[1],color=list(map(fmap,ts2)))
plt.show()