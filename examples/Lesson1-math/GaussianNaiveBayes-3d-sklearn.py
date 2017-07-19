import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation

from sklearn.naive_bayes import GaussianNB
import numpy as np
mpl.style.use('seaborn-dark')

def gaussian(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))



gnb = GaussianNB()

data = np.random.random([200,3])
data[:100,0] += 0.6
data[:100,1] += 0.6
data[:100,2] += 0.6
vali = np.ones([200])
vali[:100] = 0
GNB=gnb.fit(data, vali)

data_v = np.random.random([200,3])*2
vali_v = GNB.predict(data_v)
mk = [20, 6]
marker = [mk[int(itr)] for itr in vali]
marker_v = [mk[int(itr)] for itr in vali_v]


fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.axis('equal')
ax.scatter(data[:,0], data[:,1], data[:,2], c=marker, alpha=0.5)

fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.axis('equal')
ax.scatter(data_v[:,0], data_v[:,1], data_v[:,2], c=marker_v, alpha=0.5)

plt.show()