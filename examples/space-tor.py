import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

mpl.style.use('fivethirtyeight')


line = np.linspace(0,1,101)
tr = np.diag(np.linspace(0,1,6))
matx = np.matmul(tr,np.ones([6,101]))
maty = np.ones([6,101])*line
print(line)
def sigmoid(rt):
    return 1/(1+np.exp(-rt))
def space_tor(xx,yy):
    mx=[]
    my=[]
    for x,y in zip(xx,yy):
        mat1 = np.array([[ 15.26323795,   5.76201916],
                         [ 15.1089325,    5.76012659]])
        mat2 = np.array([[3, -2],
                         [1, 1]])
        cst1 = np.array([-7.17002916, -9.71927357])
        cst2 = np.array([-1,-1])
        re=np.matmul(np.array([x,y]),mat2)+cst2
        mx.append(re[0])
        my.append(re[1])
    return sigmoid(np.array(mx)),sigmoid(np.array(my))

for itr in range(len(matx)):
    xx1,yy1 = space_tor(matx[itr],maty[itr])
    xx2,yy2 = space_tor(maty[itr],matx[itr])
    cont=0
    for x1,x2,x3,x4 in zip(xx1,yy1,matx[itr],maty[itr]):
        if(cont%20==0):
            plt.plot([x1,x3],[x2,x4],alpha=0.1,c='k')
            plt.scatter([x1],[x2],marker='o',s=140)
            plt.scatter([x3],[x4],marker='+',s=140)
        cont+=1
    plt.plot(matx[itr],maty[itr],'--',c='r',alpha=0.5)
    plt.plot(maty[itr],matx[itr],'--',c='r',alpha=0.5)
    plt.plot(xx1,yy1,'--',c='b',alpha=0.5)
    plt.plot(xx2,yy2,'--',c='b',alpha=0.5)
plt.show()               
