import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

x = np.array([
    [1., 2.],
    [3., 4.]
])

m = np.array([
    [10., 30.],
    [20., 40.]
])

v = np.array([
    [4., 8.],
    [16., 12.]
])

eps = 2
beta1 = 0.68
beta2 = 0.27
lr = 0.45
t = 1

lrt = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

x_node = tf.Variable(x, dtype=tf.float32)
m_node = tf.Variable(m, dtype=tf.float32)
v_node = tf.Variable(v, dtype=tf.float32)
res_node = x_node - lrt * m_node / (tf.sqrt(v_node) + eps)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_res = sess.run(res_node).astype(np.float32)
tensors_saver.add(tf_res)
