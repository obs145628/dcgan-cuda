import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

Y_HAT = np.array([[[
   [ 9.,  13.],
   [  2.,  12.]],

  [[ -8.,  9.],
   [  3.,  -9.]]],


 [[[ 11.,   3.],
   [ 0., -22.]],

  [[  3.,   9.],
   [ 18.,  -8.]]]]
)

bias = np.array([1, 2])

Y = np.array([[[
   [ 7.,  11.],
   [  -1.,  4.]],

  [[ -6.,  3.],
   [  5.,  4.]]],


 [[[ 15.,   4.],
   [ -2., -22.]],

  [[  0.,   9.],
   [ 22.,  -6.]]]]
)

y_hat_node = tf.Variable(Y_HAT, dtype=tf.float32)
y_node = tf.Variable(Y, dtype=tf.float32)
bias_node = tf.Variable(bias, dtype=tf.float32)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
res = tf.nn.bias_add(y_hat_node, bias_node, data_format='NHWC')
res_tf = sess.run(res)
mse_node = tf.losses.mean_squared_error(labels=y_node, predictions=res)
mse_val = sess.run(mse_node)

db_node, dy_hat_node, d_res = tf.gradients(mse_node, [bias_node, y_hat_node, res])
db_tf = sess.run(db_node)
dy_hat_tf = sess.run(dy_hat_node)
d_res_tf = sess.run(d_res)

tensors_saver.add(d_res_tf)
tensors_saver.add(db_tf)
tensors_saver.add(dy_hat_tf)
