# -*- coding: utf-8 -*-
"""
All RNN cells are close to chaotic attractor.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import odeint

def attrack(w,t,a1,a2,a3,a4,a5,a6): 
    x, y, z = w
    return np.array([a1*(y-x)+a4*x*z,a2*x-x*z+a6*y,a3*z+x*y-a5*x*x])

t = np.arange(0, 6, 0.001) 
track = []
track.append(odeint(attrack, (1.1, 0, 0.0), t, args=(40,55,1.833,0.16,0.65,20))/300+0.5)
track.append(odeint(attrack, (4, 0, 0), t, args=(41,55,1.833,0.16,0.65,20))/300+0.5)
track.append(odeint(attrack, (6, 0, 0), t, args=(42,55,1.833,0.16,0.65,20))/300+0.5)



def update(num, data, lines, scate):
    eta=[2, 1, 0.1, 0.01, 0.1]
    rand=[0, 0, 0, 0, 5]
    for itr in range(3):
        lines[itr].set_xdata(data[itr][:num*10,0])
        lines[itr].set_ydata(data[itr][:num*10,1])
        lines[itr].set_3d_properties(data[itr][:num*10,2]-0.2)
        scate[itr].set_xdata(data[itr][num*10,0])
        scate[itr].set_ydata(data[itr][num*10,1])
        scate[itr].set_3d_properties(data[itr][num*10,2]-0.2)

mpl.style.use('fivethirtyeight')

x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
lines=[ax.plot([],[],[], alpha=0.4)[0] for itr in range(3)]
scate=[ax.plot([],[],[],'o-')[0] for itr in range(3)]

line_ani = animation.FuncAnimation(fig, update, 2000, fargs=(track, lines, scate),
                                   interval=50, blit=False)
plt.show()