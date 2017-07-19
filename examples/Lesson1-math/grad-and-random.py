# -*- coding: utf-8 -*-
"""
The gradient we used in our method is random.
Do not mix it with that we talked before.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

mpl.style.use('fivethirtyeight')
def sigmoid(x):
    return 1/(1+np.exp(-x))
def d_sigmoid(x):
    return np.exp(-x)/(1+np.exp(-x))**2

def function(A, x, y):
    v=np.array([x, y])
    return np.square(sigmoid(x+y)-sigmoid(np.dot(v,A)))


x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
A = np.random.random([2,1])*5-2.5
zs = np.array([function(A, x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
surface=ax.plot_surface(X, Y, Z, alpha=0.5)

def update(num):
    A = np.random.random([2,1])*5
    zs = np.array([function(A, x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surface=ax.plot_surface(X, Y, Z, alpha=0.5)



line_ani = animation.FuncAnimation(fig, update, 100,
                                   interval=1000, blit=False)
plt.show()