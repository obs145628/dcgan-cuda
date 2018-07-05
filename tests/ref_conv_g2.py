import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

WEIGHTS_PATH = sys.argv[1]
tensors_saver.set_out_path(sys.argv[2])

BATCH = 64

x = np.random.randn(BATCH, 16, 16, 128).astype(np.float32)
w = np.random.randn(5, 5, 64, 128).astype(np.float32)
y = np.random.randn(BATCH, 32, 32, 64).astype(np.float32)

x_node = tf.Variable(x)
w_node = tf.Variable(w)
y_node = tf.Variable(y)
yh_node = tf.nn.conv2d_transpose(x_node, w_node, y.shape, [1, 2, 2, 1], 'SAME')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

res = sess.run(yh_node)
tensors_saver.add(res)


data = tensors_saver.Saver(WEIGHTS_PATH)
data.add(x)
data.add(w)
data.save()
