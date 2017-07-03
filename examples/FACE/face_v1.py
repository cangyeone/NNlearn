from ResnetV1 import *
from loadfile import *
import numpy as numpy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt

if __name__=="__main__":
    img, vali = load()
    with tf.variable_scope("input_layer"):
        inputs = tf.placeholder(tf.float32,[20,96,96,1])
        labels = tf.placeholder(tf.float32,[20,2])
    tf.summary.image("input_layer",inputs, 20)
    outputs_pre, nets= inference(inputs, 0.8)
    tf.summary.histogram("histogram",outputs_pre)
    logits = slim.fully_connected(outputs_pre, 2, activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(1e-5),
                scope='Logits', reuse=False)
    
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            #labels=labels, logits=logits, name='cross_entropy_per_example')
    #cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    cross_entropy_mean=tf.reduce_mean(tf.square(logits-labels))
    train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy_mean)
    #tf.add_to_collection('losses', cross_entropy_mean)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    images_in=np.zeros([20,96,96,1])
    labels_in=np.zeros([20,2])
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("logdir", sess.graph)
    for itr in range(200):
        midx=np.random.choice(2140,20)

        for itry in range(20):
            images_in[itry,:,:,0]=np.reshape(img[midx[itry]],[96,96])
            labels_in[itry,:]=vali[midx[itry]][20:22]
        sess.run(train_step, feed_dict={inputs:images_in,labels:labels_in})
        if(itr%5==0):
            summary, _ = sess.run([merged, train_step],
                                        feed_dict={inputs:images_in,labels:labels_in})
            train_writer.add_summary(summary, itr)
            print(sess.run(cross_entropy_mean, feed_dict={inputs:images_in,labels:labels_in}))
        
            midx=np.random.choice(2140,20)
    
    midx=np.random.choice(2140,20)
    for itry in range(20):
        images_in[itry,:,:,0]=np.reshape(img[midx[itry]],[96,96])
        labels_in[itry,:]=vali[midx[itry]][20:22]
    out=sess.run(logits, feed_dict={inputs:images_in,labels:labels_in})
    x,y= np.mgrid[0:1:96j, 0:1:96j]
    fig=plt.figure(1)
    for itr in range(20):
        fig.clf()
        ax=fig.add_subplot(111)
        ax.contourf(x, y, np.transpose(images_in[itr,:,:,0]))
        ax.scatter([out[itr][0]],[out[itr][1]],marker='o')
        ax.scatter([labels_in[itr][0]],[labels_in[itr][1]],marker='+')
        fig.savefig("figures/i%d.jpg"%itr)
    
    