import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

def function(x, y):
    return x**2-y**2
def grad1(x, y, name='G'):
    if name=="S":
        return [-2 * x / (np.abs(x) + 0.000001), 2 * y / (np.abs(y) + 0.000001)]
    else:
        return [-2*x, +2*y]

def update(num, data, lines, scate):
    eta=[2, 0.8, 0.1, 0.1, 0.1]
    rand=[0, 0, 0, 0, 0]
    tp=["G", "G", "G", "S", "G"]
    for itr in range(5):
        data[itr][0].append(data[itr][0][-1] + eta[itr] * grad1(data[itr][0][-1], data[itr][1][-1], name=tp[itr])[0] * (rand[itr] * (np.random.random()-0.5) + 1))
        data[itr][1].append(data[itr][1][-1] + eta[itr] * grad1(data[itr][0][-1], data[itr][1][-1], name=tp[itr])[1] * (rand[itr] * (np.random.random()-0.5) + 1))
        lines[itr].set_xdata(data[itr][0])
        lines[itr].set_ydata(data[itr][1])
        z = function(np.array(data[itr][0]), np.array(data[itr][1]))
        lines[itr].set_3d_properties(z)
        scate[itr].set_ydata(data[itr][1][-1])
        scate[itr].set_xdata(data[itr][0][-1])
        scate[itr].set_3d_properties(z[-1])

mpl.style.use('fivethirtyeight')


x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
ax.plot_surface(X, Y, Z, alpha=0.5)

lines=[ax.plot([],[],[],alpha=0.5)[0] for itr in range(5)]
scate=[ax.plot([],[],[],'o-',alpha=1)[0] for itr in range(5)]
data=[[[-3],[0]],[[-3],[0]],[[-3],[0]],[[-3],[0]],[[-3],[0.1]]]

line_ani = animation.FuncAnimation(fig, update, 100, fargs=(data, lines, scate),
                                   interval=50, blit=False)
plt.show()