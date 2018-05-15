import numpy as np
import tensorflow as tf
import tensors_saver
import sys

N = 10
C = 4
F = 7
HX = 64
WX = 64

HK = 5
WK = 5

SH = 2
SW = 2
P = [1, 2, 1, 2] #top, bottom, left, right

X = np.random.randn(N, HX, WX, C)
K = np.random.randn(HK, WK, C, F)
Y = np.random.randn(N, int((HX - HK + P[0]+P[1])/SH + 1), int((WX - WK + P[2]+P[3])/SW + 1), F)

if Y.shape[1] != (HX - HK + P[0]+P[1])/SH + 1:
    print('invalid height')
    sys.exit(1)

if Y.shape[2] != (WX - WK + P[2]+P[3])/SW + 1:
    print('invalid width')
    sys.exit(1)

x_node = tf.Variable(X, dtype=tf.float32)
k_node = tf.Variable(K, dtype=tf.float32)
y_node = tf.Variable(Y, dtype=tf.float32)

out_node = tf.nn.conv2d(x_node, k_node, [1, SH, SW, 1], padding='SAME')
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
