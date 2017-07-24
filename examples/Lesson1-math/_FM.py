import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from scipy.integrate import odeint
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=100, metadata=metadata)

def attrack(w,t,a1,a2,a3,a4,a5,a6): 
    x, y, z = w
    return np.array([a1*(y-x)+a4*x*z,a2*x-x*z+a6*y,a3*z+x*y-a5*x*x])

t = np.arange(0, 6, 0.001) 
data = []
data.append(odeint(attrack, (1.1, 0, 0.0), t, args=(40,55,1.833,0.16,0.65,20))/300+0.5)
data.append(odeint(attrack, (4, 0, 0), t, args=(41,55,1.833,0.16,0.65,20))/300+0.5)
data.append(odeint(attrack, (6, 0, 0), t, args=(42,55,1.833,0.16,0.65,20))/300+0.5)

mpl.style.use('fivethirtyeight')


fig = plt.figure()
ax = fig.gca(projection='3d')
lines=[ax.plot([],[],[], alpha=0.4)[0] for itr in range(3)]
scate=[ax.plot([],[],[],'o-')[0] for itr in range(3)]




with writer.saving(fig, "writer_test.mp4", 100):
    for num in range(599):
        for itr in range(3):
            lines[itr].set_xdata(data[itr][:num*10,0])
            lines[itr].set_ydata(data[itr][:num*10,1])
            lines[itr].set_3d_properties(data[itr][:num*10,2]-0.2)
            scate[itr].set_xdata(data[itr][num*10,0])
            scate[itr].set_ydata(data[itr][num*10,1])
            scate[itr].set_3d_properties(data[itr][num*10,2]-0.2)
            writer.grab_frame()