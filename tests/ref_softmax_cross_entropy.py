import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

y = np.array([
    [0.1, 0.2, 0.7],
    [0.8, .1, .1],
    [0.1, 0.3, 0.6],
    [.6, .2, .2]
])

logits = np.array([
    [0.1, 1.2, 4.3],
    [4.1, 0.2, 7.3],
    [0.06, 2.01, 0.23],
    [5.6, 2.3, 1.18]
])

y_node = tf.Variable(y, dtype=tf.float32)
logits_node = tf.Variable(logits, dtype=tf.float32)
cross_node = tf.nn.softmax_cross_entropy_with_logits(labels=y_node, logits=logits_node)
loss_node = tf.reduce_mean(cross_node)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_loss = sess.run(loss_node)
tensors_saver.add(tf_loss)
