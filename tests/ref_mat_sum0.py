import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

x = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

x_node = tf.Variable(x, dtype=tf.float32)
y_node = tf.reduce_sum(x_node, axis=0)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_y = sess.run(y_node)
tensors_saver.add(tf_y)
