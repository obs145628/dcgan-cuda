import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

out = np.array(
[[[[ 14.,  12.],
   [  7.,  -6.]],

  [[ -5.,  -5.],
   [  9., -11.]]],


 [[[ 19.,  -1.],
   [ -2.,   2.]],

  [[  7.,  11.],
   [ 12., -11.]]]])
   
bias = np.array(
    [[[[ 2.,  2.],
   [  2.,  2.]],

  [[ -1.,  -1.],
   [  -1., -1.]]],


 [[[ 2.,  2.],
   [ 2.,   2.]],

  [[  -1.,  -1.],
   [ -1., -1.]]]]
)

out_node = tf.Variable(out, dtype=tf.float32)
bias_node = tf.Variable(bias, dtype=tf.float32)
res = tf.add(out_node, bias_node)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
res_tf = sess.run(res)
tensors_saver.add(res_tf)