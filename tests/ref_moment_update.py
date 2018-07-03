import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

a = np.array([
    [1., 2.],
    [3., 4.]
])

b = np.array([
    [10., 30.],
    [20., 40.]
])

c1 = 5.7
c2 = 3.9

a_node = tf.Variable(a, dtype=tf.float32)
b_node = tf.Variable(b, dtype=tf.float32)

res_node = c1 * a_node + c2 * b_node

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_res = sess.run(res_node).astype(np.float32)
tensors_saver.add(tf_res)
