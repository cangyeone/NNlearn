import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation

import numpy as np
mpl.style.use('seaborn-dark')

def function(data):
    u = data[0]*2.5
    v = data[1]*2.5
    return u * u + 5 * v * np.sin(v) + 5 * v
def grad1(data):
    x=data[0]*2.5
    y=data[1]*2.5
    return [-2*x, -5 * np.sin(y) + 5 * y * np.cos(y) - 5]

def Gen_RandLine(length, dims=3):
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)*3
    lineData[2, 0] = function(lineData[:2, 0])
    for index in range(1, length):
        step = np.array(grad1(lineData[:2, index - 1]))
        lineData[:2, index] = lineData[:2, index - 1] + step*0.01
        lineData[2, index] = function(lineData[:2, index])

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines
fig = plt.figure()
ax = fig.gca(projection='3d')

x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([function([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, alpha=0.5)


data = [Gen_RandLine(1000, 3) for index in range(50)]

lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

#ax.set_xlim3d([-5.0, 5.0])
#ax.set_xlabel('X')

#ax.set_ylim3d([-5.0, 5.0])
#ax.set_ylabel('Y')

#ax.set_zlim3d([-5.0, 5.0])
#ax.set_zlabel('Z')

line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   interval=50, blit=False)
plt.show()