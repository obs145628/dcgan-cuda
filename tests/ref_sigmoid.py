import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

logits = np.array([
    [0.1, 1.2, 4.3],
    [4.1, 0.2, 7.3],
    [0.06, 2.01, 0.23],
    [5.6, 2.3, 1.18]
])

logits_node = tf.Variable(logits, dtype=tf.float32)
y_hat_node = tf.nn.sigmoid(logits_node)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_y_hat = sess.run(y_hat_node)
tensors_saver.add(tf_y_hat)
