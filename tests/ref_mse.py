import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

np.random.seed(3531354)
y = np.random.rand(8000, 3)

np.random.seed(3531354)
y_hat = np.random.rand(8000, 3)

y_node = tf.Variable(y, dtype=tf.float32)
y_hat_node = tf.Variable(y_hat, dtype=tf.float32)
loss_node = tf.losses.mean_squared_error(y_node, y_hat_node)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_loss = sess.run(loss_node)
tensors_saver.add(tf_loss)
