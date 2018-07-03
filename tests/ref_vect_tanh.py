import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

features = np.array([[0.1, 1.2, -4.3, 4.1, -0.2, 7.3, 0.06, 2.01, 0.23, 5.6, 2.3, 1.18]])

y_hat_node = tf.nn.tanh(features)
sess = tf.Session()
tf_y_hat = sess.run(y_hat_node)
tensors_saver.add(tf_y_hat.astype(np.float32))
