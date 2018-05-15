import numpy as np
import tensorflow as tf
import tensors_saver
import sys

HX = 47
WX = 28

HK = 12
WK = 7

SH = 5
SW = 3


X = np.random.randn(1, HX, WX, 1)
K = np.random.randn(HK, WK, 1, 1)
Y = np.random.randn(1, int((HX - HK)/SH + 1), int((WX - WK)/SW + 1), 1)

if Y.shape[1] != (HX - HK)/SH + 1:
    print('invalid height')
    sys.exit(1)

if Y.shape[2] != (WX - WK)/SW + 1:
    print('invalid width')
    sys.exit(1)

x_node = tf.Variable(X, dtype=tf.float32)
k_node = tf.Variable(K, dtype=tf.float32)
y_node = tf.Variable(Y, dtype=tf.float32)

out_node = tf.nn.conv2d(x_node, k_node, [1, SH, SW, 1], padding='VALID')
loss_node = tf.losses.mean_squared_error(tf.layers.flatten(y_node), tf.layers.flatten(out_node))
dx_node, dk_node, dy_node = tf.gradients(loss_node, [x_node, k_node, out_node])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

out = sess.run(out_node)
dx = sess.run(dx_node)
dk = sess.run(dk_node)
dy = sess.run(dy_node)


tensors_saver.save_all('conv_in.npz', [X, K, dy])
tensors_saver.save_all('out.npz', [out, dk, dx])

print(out.shape)
print(y_node.get_shape())
