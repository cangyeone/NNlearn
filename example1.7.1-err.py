# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:23:46 2017

@author: Cangye@hotmail.com
"""

import numpy as np


class GenTestData():
    def __init__(self,shape):
        self.shape=shape
    def func(self,dt):
        if(dt[0]+dt[1]<1):
            rt=[0.1]
        #elif((dt[0]+dt[1])>1.5):
            #rt=[0.1]
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


class BPAlg():
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def d_sigmiod(self,x):
        return np.exp(-x)/(1+np.exp(-x))**2
    def __init__(self,shape):
        self.shape=shape
        self.layer=len(shape)
        self.W=[]
        self.b=[]
        self.e=[]
        self.y=[]
        self.dW=[]
        self.v=[]
        self.db=[]
        self.d_sigmoid_v=[]
        for itrn in range(self.layer-1):
            self.W.append(np.ones([shape[itrn],shape[itrn+1]]))
            self.dW.append(np.ones([shape[itrn],shape[itrn+1]]))
            self.b.append(np.ones([shape[itrn+1]]))
            self.db.append(np.ones([shape[itrn+1]]))
        for itr in shape:
            self.e.append(np.ones([itr]))
            self.y.append(np.ones([itr]))
            self.v.append(np.ones([itr]))
            self.d_sigmoid_v.append(np.ones([itr]))
    def forward(self,data):
        self.y[0][:]=np.average(data,axis=0)
        temp_y=data
        for itrn in range(self.layer-1):
            temp_y=np.dot(temp_y,self.W[itrn])
            temp_b=self.b[itrn]
            temp_v=np.add(temp_y,temp_b)
            temp_y=self.sigmoid(temp_v)
            self.y[itrn+1][:]=np.average(temp_y,axis=0)
            self.d_sigmoid_v[itrn+1][:]=np.average(self.d_sigmiod(temp_v),axis=0)
        return self.y[-1]
    def back_forward(self,dest):
        self.e[self.layer-1]=np.average(dest,axis=0)-self.y[self.layer-1]
        temp_delta=self.e[self.layer-1]*self.d_sigmoid_v[self.layer-1]
        temp_delta=np.reshape(temp_delta,[-1,1])
        self.dW[self.layer-2][:]=np.dot(np.reshape(self.y[self.layer-2],[-1,1]),np.transpose(temp_delta))
        self.db[self.layer-2][:]=np.transpose(temp_delta)
        #print(self.dW[self.layer-2])
        for itrn in range(self.layer-2,0,-1):
            sigma_temp_delta=np.dot(self.W[itrn],temp_delta)
            temp_delta=sigma_temp_delta*np.reshape(self.d_sigmoid_v[itrn],[-1,1])
            self.dW[itrn-1][:]=np.dot(np.reshape(self.y[itrn-1],[-1,1]),np.transpose(temp_delta))
            self.db[itrn-1][:]=np.transpose(temp_delta)
    def data_feed(self,data,dest,eta):
        self.forward(data[0])
        self.back_forward(dest[0])
        for itrn in range(self.layer-1):
            self.W[itrn][:]=self.W[itrn]+eta*self.dW[itrn]
            self.b[itrn][:]=self.b[itrn]+eta*self.db[itrn]
        #print("===============================")
        #print(self.W)
        #print(self.dW)
dt=GenTestData([50,2])
tsc=BPAlg([2,2,1])
for itrn in range(2000):
    data=dt.GenData()
    vali=dt.GenVali()
    tsc.data_feed(data,vali,4)
#print(tsc.forward(np.array([[1,1]])))   

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
            l1=tsc.forward([X[ity,itx],Y[ity,itx]])
            Z[ity,itx]=l1
    return Z
x=np.linspace(0,1,100)
y=np.linspace(0,1,100)
X,Y=np.meshgrid(x,y)
Z=GenZ(X,Y)
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,rstride=8,cstride=8, alpha=0.3)
ax.contour(X,Y,Z,zdir='z',offset=0, cmap=plt.cm.coolwarm)
plt.show()
