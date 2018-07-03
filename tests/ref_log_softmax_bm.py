import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

np.random.seed(3531354)
logits = np.random.rand(8000, 3)

logits_node = tf.Variable(logits, dtype=tf.float32)
y_hat_node = tf.nn.log_softmax(logits_node)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_y_hat = sess.run(y_hat_node)
tensors_saver.add(tf_y_hat)
