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

x_node = tf.Variable(x, dtype=tf.float32)
w_node = tf.Variable(w, dtype=tf.float32)
b_node = tf.Variable(b, dtype=tf.float32)
res_node = tf.matmul(x_node, w_node) + b_node
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_res = sess.run(res_node)
tensors_saver.add(tf_res.astype(np.float32))
