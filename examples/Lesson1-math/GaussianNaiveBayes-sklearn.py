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

data = np.random.random([200,1])
data[:100] += 0.7
vali = np.ones([200])
vali[:100] = 0
GNB=gnb.fit(data, vali)

data_v = np.random.random([200,1])*2
vali_v = GNB.predict(data_v)
mk = [20, 6]
print(vali_v)
marker = [mk[int(itr)] for itr in vali_v]

x=np.linspace(-1,2,100)
plt.plot(x,gaussian(x, 0.35, 0.35))
plt.plot(x,gaussian(x, 1.20, 0.35))
plt.text(-1, 0.4, "gaussian1")
plt.text(2, 0.4, "gaussian2")
plt.scatter(data_v, np.zeros([200]), c=marker, s=20, marker='o')
plt.show()
