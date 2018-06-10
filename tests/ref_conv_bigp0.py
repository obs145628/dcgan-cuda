import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver
import xorshift

tensors_saver.set_out_path(sys.argv[1])
xorshift.seed(234)

BATCH = 32

x = xorshift.np_f32((BATCH, 77, 77, 8))
w = xorshift.np_f32((5, 5, 8, 64))

x_node = tf.Variable(x)
w_node = tf.Variable(w)
y_node = tf.nn.conv2d(x_node, w_node, [1, 2, 2, 1], 'VALID')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y = sess.run(y_node)
tensors_saver.add(y)
