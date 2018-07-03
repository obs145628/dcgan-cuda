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
y_hat_node = tf.nn.relu(x_node)
loss_node = tf.losses.mean_squared_error(y_node, y_hat_node)
dx_node, dy_hat_node = tf.gradients(loss_node, [x_node, y_hat_node])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_y_hat = sess.run(y_hat_node)
tf_loss = sess.run(loss_node)
tf_dx = sess.run(dx_node)
tf_dy_hat = sess.run(dy_hat_node)
tensors_saver.add(tf_y_hat)
tensors_saver.add(tf_loss)
tensors_saver.add(tf_dx)
tensors_saver.add(tf_dy_hat)
