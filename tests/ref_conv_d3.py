import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

WEIGHTS_PATH = sys.argv[1]
tensors_saver.set_out_path(sys.argv[2])

BATCH = 64

x = np.random.randn(BATCH, 8, 8, 256).astype(np.float32)
w = np.random.randn(5, 5, 256, 512).astype(np.float32)

x_node = tf.Variable(x)
w_node = tf.Variable(w)
y_node = tf.nn.conv2d(x_node, w_node, [1, 2, 2, 1], 'SAME')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y = sess.run(y_node)
tensors_saver.add(y)

data = tensors_saver.Saver(WEIGHTS_PATH)
data.add(x)
data.add(w)
data.save()
