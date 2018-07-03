import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

np.random.seed(3531354)
x = np.random.rand(8000, 3)

np.random.seed(3531354)
y = np.random.rand(8000, 3)

x_node = tf.Variable(x, dtype=tf.float32)
y_node = tf.Variable(y, dtype=tf.float32)
cross_node = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_node, logits=x_node)
loss_node = tf.reduce_mean(cross_node)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_loss = sess.run(loss_node)
tensors_saver.add(tf_loss)
