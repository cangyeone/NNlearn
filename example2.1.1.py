import tensorflow as tf
import numpy as np


X=tf.placeholder(dtype=tf.float32,shape=[4,1])
Y=tf.placeholder(dtype=tf.float32,shape=[4,1])

A=tf.Variable(tf.zeros([4,4]))

C=tf.matmul(A,X)
sess=tf.Session()

ce = tf.reduce_sum(tf.abs(Y-C))
ts = tf.train.AdamOptimizer(1e-2).minimize(ce)

lstm=tf.contrib.rnn.LSTMCell(3)


init=tf.global_variables_initializer()

#注意，此处变量初始化一定要在所有变量之后！
sess.run(init)

for itr in range(1000):
    inX=np.random.random([4,1])
    simA=np.ones([4,4])
    inY=np.matmul(simA,inX)
    #上部分用于生成数据
    sess.run(ts,feed_dict={X:inX,Y:inY})
CL=A.value()
print(sess.run(CL))
print(sess.run(lstm(A,False)))

