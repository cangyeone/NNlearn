from ResnetV1 import *
from loadfile import *
import numpy as numpy
import tensorflow as tf

if __name__=="__main__":
    #img, vali = load()
    with tf.variable_scope("input_layer"):
        inputs = tf.get_variable("image",[20,96,96,1],dtype=tf.float32)
        labels = tf.get_variable("label",[20,2],dtype=tf.float32)
    outputs_pre, _= inception_resnet_v1(inputs, is_training=True,
                                dropout_keep_prob=0.8,
                                bottleneck_layer_size=64,
                                reuse=None, 
                                scope='IRV1')
    
    logits = slim.fully_connected(outputs_pre, 2, activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(1e-5),
                scope='Logits', reuse=False)
    
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    cross_entropy_new = -tf.reduce_sum(labels*tf.log(logits))
    train_step = tf.train.RMSPropOptimizer(0.5, decay=0.9, momentum=0.9, epsilon=1.0).minimize(cross_entropy_mean)
    tf.add_to_collection('losses', cross_entropy_mean)
    images_in=np.ones([20,96,96,1])
    labels_in=np.ones([20,2])
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    for itr in range(200):
        midx=np.random.choice(2140,20)
        """
        for itry in range(20):
            images_in[itry,:,:,0]=np.reshape(img[midx[itry]],[96,96])
            labels_in[itry,:]=vali[midx[itry]][20:22]
        sess.run(train_step, feed_dict={inputs:images_in,labels:labels_in})
        """

        print(sess.run(train_step, feed_dict={inputs:images_in,labels:labels_in}))
        break