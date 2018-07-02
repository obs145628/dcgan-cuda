import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

np.random.seed(3531354)
x = np.random.rand(8000, 3)

np.random.seed(3531354)
y = np.random.rand(8000, 3)

x_node = tf.Variable(x)
y_node = tf.Variable(y)
loss_node = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_node, logits=x_node))
dx_node = tf.gradients(loss_node, [x_node])[0]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_loss = sess.run(loss_node)
tf_dx = sess.run(dx_node)
tensors_saver.add(tf_loss.astype(np.float32))
tensors_saver.add(tf_dx.astype(np.float32))
