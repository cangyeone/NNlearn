import tensorflow as tf
import numpy as np    

x = tf.placeholder(dtype=tf.float32,shape=[1,8])
y = tf.placeholder(dtype=tf.float32,shape=[1,8])
x2 = tf.pow(x,2)
x3 = tf.pow(x,3)
x4 = tf.pow(x,4)
x5 = tf.pow(x,5)
x6 = tf.pow(x,6)
x7 = tf.pow(x,7)
x8 = tf.pow(x,8)
x_v = tf.concat([x,x2,x3,x4,x5,x6,x7,x8],axis=0)
A = tf.Variable(tf.ones([1,8]))
b = tf.Variable(tf.ones([1,1]))
bbc = tf.concat([b,b,b,b,b,b,b,b],axis=1)
y_new = tf.matmul(A,x_v)+b
loss = tf.reduce_mean(tf.square(y-y_new))

train_step = tf.train.AdamOptimizer(1e1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for itr in range(10000):
    sess.run(train_step, feed_dict={x:(np.array([[0,0.2,0.3,0.4,0.5,0.6,0.7,0.8]])),
                                    y:(np.array([[0,0.15,0.35,0.4,0.6,0.5,0.75,0.8]]))})

print(sess.run(A.value(), feed_dict={x:(np.array([[0,2,3,4,5,6,7,8]])),
                                    y:(np.array([[0,1.5,3.5,4,6,5,7.5,8]]))}))
def function(a):
    xx=np.concatenate([a,a**2,a**3,a**4,a**5,a**6,a**7,a**8],axis=0)
    ma=np.array(sess.run(A.value()))
    mb=np.array(sess.run(b.value()))
    return np.matmul(ma,xx)+mb[0,0]
import matplotlib.pyplot as plt
lin=np.array([np.linspace(0,0.9,100)])
ly =function(lin)
plt.plot(lin[0],ly[0])
plt.scatter([0,0.2,0.3,0.4,0.5,0.6,0.7,0.8],[0,0.15,0.35,0.4,0.6,0.5,0.75,0.8])
plt.show()
