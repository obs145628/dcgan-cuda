import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

x = np.array([
    [1, 0.5, 0.4],
    [0.2, 0.1, 0.7],
    [2.3, 2.5, 8.4],
    [1.9, 1.2, 1.4],
    [0.23, -1.6, 1.4]
])

y = np.array([
    [0.1, 0.5, 0.4],
    [0.2, 0.1, 0.7],
    [0.8, 0.05, 0.15],
    [0.3, 0.6, 0.1],
    [0.7, 0.1, 0.2]
])

x_node = tf.Variable(x)
y_node = tf.Variable(y)
loss_node = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_node, logits=x_node))
dx_node = tf.gradients(loss_node, [x_node])[0]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_loss = sess.run(loss_node)
tf_dx = sess.run(dx_node)
tensors_saver.add(tf_loss.astype(np.float32))
tensors_saver.add(tf_dx.astype(np.float32))
