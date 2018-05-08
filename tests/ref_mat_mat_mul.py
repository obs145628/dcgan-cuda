import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

a = np.array([
    [1., 2, 4],
    [4.1, 0.5, 7],
    [2, 2, 8],
    [5, 2.3, 1.1]
])

b = np.array([
    [1., 5.],
    [2., 4],
    [3, 8]
])

a_node = tf.Variable(a, dtype=tf.float32)
b_node = tf.Variable(b, dtype=tf.float32)
res_node = tf.matmul(a, b)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_res = sess.run(res_node)
tensors_saver.add(tf_res.astype(np.float32))
