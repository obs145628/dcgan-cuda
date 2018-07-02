import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

np.random.seed(3531354)
features = np.random.rand(1, 8000)

y_hat_node = tf.nn.relu(features)
sess = tf.Session()
tf_y_hat = sess.run(y_hat_node)
tensors_saver.add(tf_y_hat.astype(np.float32))
