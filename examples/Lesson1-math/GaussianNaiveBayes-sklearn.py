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
marker = [mk[int(itr)] for itr in vali_v]

x=np.linspace(-1,3,100)
plt.plot(x,gaussian(x, 0.35, 0.35))
plt.plot(x,gaussian(x, 1.20, 0.35))
plt.text(-1, 0.4, r"$exp({\frac{1}{\sqrt{2\pi}\sigma_1}\frac{-(x-\mu_1)^2}{2\sigma_1^2}})$")
plt.text(2, 0.4, r"$exp({\frac{1}{\sqrt{2\pi}\sigma_2}\frac{-(x-\mu_2)^2}{2\sigma_2^2}})$")
plt.scatter(data_v, np.zeros([200]), c=marker, s=20, marker='o')
plt.show()
