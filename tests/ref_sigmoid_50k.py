import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

N = 53147
x = np.linspace(0.0, 1.0, N)

x_node = tf.Variable(x, dtype=tf.float32)
y_node = tf.nn.sigmoid(x_node)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_y = sess.run(y_node)
tensors_saver.add(tf_y)
