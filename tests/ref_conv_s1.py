import os
import sys
import time

import numpy as np
import tensorflow as tf

BATCH = 10

x = np.random.randn(BATCH, 64, 64, 3).astype(np.float32)
w0 = np.random.randn(5, 5, 3, 64).astype(np.float32)
w1 = np.random.randn(5, 5, 64, 128).astype(np.float32)
w2 = np.random.randn(5, 5, 128, 256).astype(np.float32)
w3 = np.random.randn(5, 5, 256, 512).astype(np.float32)

x_node = tf.placeholder(tf.float32, (BATCH, 64, 64, 3))
w0_node = tf.Variable(w0)
w1_node = tf.Variable(w1)
w2_node = tf.Variable(w2)
w3_node = tf.Variable(w3)

y_node = tf.nn.conv2d(x_node, w0_node, [1, 2, 2, 1], 'SAME')
y_node = tf.nn.conv2d(y_node, w1_node, [1, 2, 2, 1], 'SAME')
y_node = tf.nn.conv2d(y_node, w2_node, [1, 2, 2, 1], 'SAME')
y_node = tf.nn.conv2d(y_node, w3_node, [1, 2, 2, 1], 'SAME')

print(y_node.get_shape())

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

total_dur = 0

for i in range(1, 10000):

    start = time.time()
    y = sess.run(y_node, feed_dict={x_node:x})
    dur = (time.time() - start) * 1000
    total_dur += dur
    print('Time = {}ms'.format(total_dur / i))
