import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

y_hat = np.array([
    [0.1, 0.2, 0.7],
    [0.8, .1, .1],
    [0.1, 0.3, 0.6],
    [.6, .2, .2],
    [.1, .1, .8],
    [.2, .3, .5],
    [.7, .1, .2],
    [.4, .3, .3],
    [.2, .1, .7],
    [.8, .1, .1]
])

y = np.array([
    [0., 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0]
])

y_node = tf.Variable(y, dtype=tf.float32)
y_hat_node = tf.Variable(y_hat, dtype=tf.float32)

acc_node = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)), tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_acc = sess.run(acc_node).astype(np.float32)
tensors_saver.add(tf_acc)
