import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

np.random.seed(3531354)
a = np.random.rand(8000, 2)

np.random.seed(3531354)
b = np.random.rand(8000, 2)

coeff = 5.7

a_node = tf.Variable(a, dtype=tf.float32)
b_node = tf.Variable(b, dtype=tf.float32)

res_node = a_node + coeff * b_node

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_res = sess.run(res_node).astype(np.float32)
tensors_saver.add(tf_res)
