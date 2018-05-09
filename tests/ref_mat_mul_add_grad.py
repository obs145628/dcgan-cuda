import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

x = np.array([
    [1., 2, 4],
    [4.1, 0.5, 7],
    [2, 2, 8],
    [5, 2.3, 1.1]
])

w = np.array([
    [1., 5.],
    [2., 4],
    [3, 8]
])

b = np.array([0.5, -4.6])

y = np.array([
    [0.1, 1.2],
    [4.1, 0.2],
    [0.06, 2.01],
    [5.6, 2.3]
])

x_node = tf.Variable(x, dtype=tf.float32)
w_node = tf.Variable(w, dtype=tf.float32)
b_node = tf.Variable(b, dtype=tf.float32)
y_hat_node = tf.matmul(x_node, w_node) + b_node
y_node = tf.Variable(y, dtype=tf.float32)
loss_node = tf.losses.mean_squared_error(y_node, y_hat_node)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


dx_node, dw_node, db_node, dy_hat_node = tf.gradients(loss_node,
                                                      [x_node, w_node, b_node, y_hat_node])


tf_dx = sess.run(dx_node)
tf_dw = sess.run(dw_node)
tf_db = sess.run(db_node)
tf_dy_hat = sess.run(dy_hat_node)

tensors_saver.add(tf_dx.astype(np.float32))
tensors_saver.add(tf_dw.astype(np.float32))
#tensors_saver.add(tf_db.astype(np.float32))
tensors_saver.add(tf_dy_hat.astype(np.float32))
