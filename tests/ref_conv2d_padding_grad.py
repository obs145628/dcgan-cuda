import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

input = tf.constant([[
    [
        [5, 0, 4],
        [-1, -1, 8],
        [2, -2, 9],
        [3, 4, -5],
        [8, -4, 1]
    ],[
        [3, 3, 1],
        [6, 2, -5],
        [3, 4, -2],
        [4, 0, 10],
        [1, 2, -3]
    ],[
        [1, 0, 5],
        [0, -2, 2],
        [0, 1, -7],
        [2, 0, -4],
        [-3, -10, 3]
    ]
   ],[
    [
        [8, 2, -4],
        [0, 1, -1],
        [4, -2, -8],
        [1, -1, 2],
        [-7, -4, -5]
    ],[
        [3, 0, 4],
        [4, 2, 9],
        [3, 0, -12],
        [-3, 0, 4],
        [6, -10, 12]
    ],[
        [1, 0, 5],
        [-8, 7, 0],
        [0, 1, 4],
        [0, -9, 1],
        [0, 1, 0]
    ]
   ]
], dtype=tf.float32, name='input')

kernel = tf.constant([
    [
        [
            [1, 1],
            [-1, 1],
            [0, 0]
        ],
        [
            [0, -1],
            [-1, 1],
            [1, 0]
        ],
        [
            [-1, -1],
            [1, 0],
            [0, 1]
        ]
    ] ,[
        [
            [0, -1],
            [0, 0],
            [1, -1]
        ],
        [
            [1, 1],
            [-1, 0],
            [1, -1]
        ],
        [
            [-1, 0],
            [-1, 0],
            [-1, 1]
        ]
    ]
], dtype=tf.float32, name='kernel')


Y = np.array([[[
   [ 9.,  13.],
   [  2.,  12.],
   [  1.,  -5.]],

  [[ -8.,  9.],
   [  3.,  -9.],
   [  4.,  10.]]],


 [[[ 11.,   3.],
   [ 0., -22.],
   [  -4.,  0.]],

  [[  3.,   9.],
   [ 18.,  -8.],
   [  9.,  1.]]]]
)

x_node = tf.Variable(input, dtype=tf.float32)
k_node = tf.Variable(kernel, dtype=tf.float32)
y_node = tf.Variable(Y, dtype=tf.float32)

y_hat_node = tf.nn.conv2d(x_node, k_node, [1, 2, 2, 1], "SAME")

loss_node = tf.losses.mean_squared_error(y_node, y_hat_node)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


dx_node, dk_node, dy_hat_node = tf.gradients(loss_node,
                                             [x_node, k_node, y_hat_node])


tf_dx = sess.run(dx_node)
tf_dk = sess.run(dk_node)
tf_dy_hat = sess.run(dy_hat_node)

tensors_saver.add(tf_dx.astype(np.float32))
tensors_saver.add(tf_dk.astype(np.float32))
tensors_saver.add(tf_dy_hat.astype(np.float32))
