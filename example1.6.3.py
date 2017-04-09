# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:16:29 2017

@author: Cangye@hotmail.com
"""

import numpy as np
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
            x=X[ity,itx]
            y=Y[ity,itx]
            l1=(x-0.5)**2+(y-0.5)**2
            l1f=sigmoid(l1)
            Z[ity,itx]=l1f
    return Z
x=np.linspace(0,1,100)
y=np.linspace(0,1,100)
X,Y=np.meshgrid(x,y)
Z=GenZ(X,Y)
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,rstride=8,cstride=8, alpha=0.3)
ax.contour(X,Y,Z,zdir='z',offset=0.5, cmap=plt.cm.coolwarm)
plt.show()

