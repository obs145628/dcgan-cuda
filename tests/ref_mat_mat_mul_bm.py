import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

np.random.seed(3531354)
a = np.random.rand(8000, 3)

np.random.seed(3531354)
b = np.random.rand(8000, 3).reshape(3, 8000)

a_node = tf.Variable(a, dtype=tf.float32)
b_node = tf.Variable(b, dtype=tf.float32)
res_node = tf.matmul(a, b)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_res = sess.run(res_node)
tensors_saver.add(tf_res.astype(np.float32))
