import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

a = np.array([
    [0.1, 1.2, 4.3],
    [4.1, 0.2, 7.3],
    [0.06, 2.01, 0.23],
    [5.6, 2.3, 1.18]
])

b = np.array([
    [2.1, -5, 4.2],
    [4.1, 0.6, 8.2],
    [102.76, 342.91, -2.23],
    [-50.6, 56.23, 15.18]
])

a_node = tf.Variable(a, dtype=tf.float32)
b_node = tf.Variable(b, dtype=tf.float32)
y_node = a_node + b_node
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_y = sess.run(y_node)
tensors_saver.add(tf_y)
