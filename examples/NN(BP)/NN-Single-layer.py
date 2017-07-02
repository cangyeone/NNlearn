# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:16:29 2017

@author: Cangye@hotmail.com
"""


import tensorflow as tf
import numpy as np
#产生测试数据
class GenTestData():
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
        #for itr in range(len(self.data)):
            #self.data[itr,2]=1
        return self.data
#最小二乘方法
class LMS():
    def Sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def DSigmiod(self,x):
        return np.exp(-x)/(1+np.exp(-x))**2
    def __init__(self,shape=[2,1]):
        self.shape=shape
        self.W=np.random.random(shape)
        self.b=np.random.random(shape[1])
    def train(self,data,vali,eta):
        nu=np.dot(data,self.W)+np.tile(self.b,np.shape(vali))
        para=(vali-self.Sigmoid(nu))*self.DSigmiod(nu)
        para=np.reshape(para,[-1])
        x=np.transpose(data)
        grad_t=np.multiply(x,para)
        grad=np.transpose(grad_t)
        grad_ave=np.average(grad,axis=0)
        grad_ave=np.reshape(grad_ave,self.shape)
        self.W=np.add(self.W,eta*grad_ave)
        self.b=np.add(self.b,eta*np.average(para))
    def valid(self,data):
        return self.Sigmoid(np.dot(data,self.W))
#产生训练数据
genData=GenTestData([50,2])
lms=LMS([2,1])
#迭代训练
for itr in range(6000):
    test_data=genData.GenData()
    test_vali=genData.GenVali()
    lms.train(test_data,test_vali,2)
#之后用于输出结果
print(lms.W)
print(lms.b)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.style.use('seaborn-darkgrid')
fig=plt.figure(1)
def sigmoid(rt):
    return 1/(1+np.exp(-rt))
def GenZ(X,Y):
    Z=np.zeros(np.shape(X))
    for ity in range(len(X)):
        for itx in range(len(X[0])):
            l1=np.matmul([X[ity,itx],Y[ity,itx]],lms.W[0:2])
            l1f=sigmoid(l1+lms.b)
            Z[ity,itx]=l1f[0]
    return Z
x=np.linspace(0,1,100)
y=np.linspace(0,1,100)
X,Y=np.meshgrid(x,y)
Z=GenZ(X,Y)
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,rstride=8,cstride=8, alpha=0.3)
ax.contour(X,Y,Z,zdir='z',offset=0, cmap=plt.cm.coolwarm)
plt.show()

