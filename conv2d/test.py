import numpy as np
import tensorflow as tf
import tensors_saver

X = np.random.randn(10, 23, 18, 3)
K = np.random.randn(7, 12, 3, 16)

SH = 2
SW = 3

x_node = tf.Variable(X, dtype=tf.float32)
k_node = tf.Variable(K, dtype=tf.float32)
out_node = tf.nn.conv2d(x_node, k_node, [1, SH, SW, 1], padding='SAME')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tensors_saver.save_all('conv_in.npz', [X, K])

out = sess.run(out_node)
tensors_saver.save_all('conv_out.npz', [out])
