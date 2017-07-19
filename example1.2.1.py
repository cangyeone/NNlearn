# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:19:44 2017

@author: Cangye@hotmail.com
"""

import tensorflow as tf
import numpy as np

class GenData():
    def __init__(self,shape):
        self.shape=shape
    def Func(self,x):
        x=x[0]
        return [2*x*x+0.5*x+0.1]
    def GenWave(self):
        self.data=np.random.random(self.shape)*2
        self.outdata=np.array(list(map(self.Func,self.data)))
        return self.data,self.outdata

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

W = tf.Variable(tf.ones([1.,1.]))
b = tf.Variable(tf.ones([1.]))
result =tf.nn.sigmoid(tf.matmul(x, W) + b)
amp=tf.Variable(tf.ones([1.]))
result=result*amp

ce = tf.reduce_mean(tf.square(result-y))
train_step = tf.train.AdamOptimizer(1e0).minimize(ce)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
dataGenerator=GenData([50,1])

for _ in range(10000):
    dt,odt=dataGenerator.GenWave()
    sess.run(train_step, feed_dict={x: dt, y: odt})

dt,odt=dataGenerator.GenWave()
rdt=sess.run(result,feed_dict={x: dt, y: odt})
print("W and b is : %f,%f"%(sess.run(W.value()),sess.run(b.value())))

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.xkcd()
mpl.style.use('seaborn-colorblind')
cof = np.polyfit(np.reshape(dt,[-1]),np.reshape(rdt,[-1]),2) 
p=np.poly1d(cof)


fig,ax=plt.subplots()

ax.scatter(np.reshape(dt,[-1]),np.reshape(odt,[-1]),marker='+')
ax.scatter(np.reshape(dt,[-1]),np.reshape(rdt,[-1]),marker='o')
ax.text(0,0.1,'$f(x)=%f+%fx+%fx^2$'%(cof[2],cof[1],cof[0]))
x=np.linspace(0,2,100)
ax.plot(x,p(x),lw=2)
ax.text(0,4,'')
plt.show()
    
  

['bmh',
 'grayscale',
 'fivethirtyeight',
 'seaborn-muted',
 'seaborn-darkgrid',
 'classic',
 'seaborn-pastel',
 'seaborn-whitegrid',
 'seaborn-colorblind',
 'seaborn-deep',
 'seaborn-poster',
 'seaborn-notebook',
 'dark_background',
 'seaborn-talk',
 'ggplot',
 'seaborn-dark',
 'seaborn-white',
 'seaborn-bright',
 'seaborn-dark-palette',
 'seaborn-paper',
 'seaborn-ticks']
















