import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

input = np.array([[
    [
        [5, 0, 4],
        [-1, -1, 8],
        [2, -2, 9],
        [3, 4, -5]
    ],[
        [3, 3, 1],
        [6, 2, -5],
        [3, 4, -2],
        [4, 0, 10]
    ],[
        [1, 0, 5],
        [0, -2, 2],
        [0, 1, -7],
        [2, 0, -4]
    ],[
        [3, 6, -1],
        [-2, 7, 0],
        [-2, 1, 4],
        [1, 0, 9]
    ]
   ],[
    [
        [8, 2, -4],
        [0, 1, -1],
        [4, -2, -8],
        [1, -1, 2]
    ],[
        [3, 0, 4],
        [4, 2, 9],
        [3, 0, -12],
        [-3, 0, 4]
    ],[
        [1, 0, 5],
        [-8, 7, 0],
        [0, 1, 4],
        [0, -9, 1]
    ],[
        [3, 0, 4],
        [2, -7, 0],
        [-2, 1, 3],
        [-1, 0, 1]
    ]
   ]
])
w = np.array([
    [
        [
            [1, -1, 0],
            [1, 1, 0]
        ],
        [
            [0, -1, 1],
            [-1, 1, 0]
        ]
    ] ,[
        [
            [0, 0, 1],
            [1, 0, -1]
        ],
        [
            [1, -1, 1],
            [-1, 0, -1]
        ]
    ]
])

Y = np.array([[[
   [  4.,   7.],
   [  2.,  -5.],
   [  1.,  2.],
   [  -9.,   3.],
   [  4.,   2.],
   [ -11.,  4.],
   [ 1.,   7.],
   [ -5.,   1.]],

  [[  4.4,   1.],
   [  -9.,  -9.],
   [  8.,  -9.],
   [  8.5,  7.],
   [  -9.,  -7.],
   [ -13., 11.],
   [ 5.,   10.],
   [ -6.,   2.]],

  [[  0.,   2.],
   [ -2.,   1.],
   [  4.,   -8.],
   [ 0. , -4.],
   [ -1.,   -7.],
   [ -6.,   -1.],
   [  3.,   0.],
   [ 5. , -2.]],

  [[  -1.,   2.],
   [  -1.,  -4.],
   [ -5. , -11.],
   [ 1.  ,1.],
   [ 2.  , 5.],
   [ -3. , -1.],
   [ 13. , -6.],
   [ 11. ,-1.]],

  [[  1. ,  1.],
   [  -5.,  -1.],
   [ -2. , -2.],
   [  -1.,  -2.],
   [ 1.  , 3.],
   [ 8.  , 2.],
   [  1. ,  -2.],
   [ -2. , -2.]],

  [[  5. , -4.],
   [  3. , -6.],
   [  2. , 2.],
   [  5. , 2.],
   [ -7. ,  7.],
   [ 8.  , -7.],
   [ 4.  , -6.],
   [ -7. ,  8.]],

  [[ -3. ,  9.],
   [ 7.  , 3.],
   [ 9.  , 0.],
   [ 0.  , 0.],
   [ -3. , -1.],
   [  0. ,  3.],
   [  -1.,   10.],
   [  -9.,  -2.]],

  [[ -1. ,  4.],
   [ -3. , -2.],
   [  2. , 2.],
   [ 9.  , 2.],
   [ - 4.,  6.],
   [  -1.,  2.],
   [  -9.,  -4.],
   [ 11. ,-1.]]],


 [[[  6.,  12.],
   [ 6.,  -6.],
   [ -1.,   -1.],
   [ 2.,   0.],
   [  -6.,   2.],
   [ -0.,  -6.],
   [  2.,   0.],
   [  -3.,  2.]],

  [[  3.,   3.],
   [  4.,  -3.],
   [  2.,   6.],
   [  7.,  -2.],
   [  3.,   3.],
   [-12.,  -3.],
   [ -3.,  -3.],
   [  4.,   3.]],
  
  [[ -4.,  12.],
   [  2.,  -4.],
   [ -1.,   1.],
   [ -2.,   1.],
   [ -8.,  12.],
   [ -2.,   4.],
   [  2.,  -1.],
   [  4.,  -3.]],

  [[  4.,  -1.],
   [  7.,  -7.],
   [  9.,  -5.],
   [ 11., -13.],
   [-12.,  15.],
   [ -9.,   9.],
   [  4.,  -7.],
   [  1.,  -1.]],

  [[  3.,   3.],
   [  4.,  -3.],
   [  9.,  -5.],
   [  7.,  -9.],
   [ -3.,  -1.],
   [  2.,   3.],
   [ -1.,  -1.],
   [  1.,   1.]],

  [[  4.,  -1.],
   [  7.,  -7.],
   [  0.,   2.],
   [  9.,  -2.],
   [  3.,  -5.],
   [  0.,  -1.],
   [  1.,  -2.],
   [  0.,   0.]],
  
  [[  1.,   1.],
   [  5.,  -1.],
   [-15.,  -1.],
   [ -7.,  15.],
   [ -1.,   1.],
   [  3.,   1.],
   [  9.,  -9.],
   [ 10.,  -9.]],

  [[  5.,  -4.],
   [  6.,  -6.],
   [  0.,  -8.],
   [-15.,   8.],
   [  4.,  -4.],
   [  3.,  -4.],
   [  1.,  -1.],
   [ 10.,  -1.]]
 ]])

input_node = tf.Variable(input, dtype=tf.float32)
w_node = tf.Variable(w, dtype=tf.float32)
y_node = tf.Variable(Y, dtype=tf.float32)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

y_hat_node = tf.nn.conv2d_transpose(input_node, w_node, [2, 8, 8, 2], [1, 2, 2, 1], "VALID")
y_hat_tf = sess.run(y_hat_node)
mse_node = tf.losses.mean_squared_error(labels=y_node, predictions=y_hat_node)
mse_val = sess.run(mse_node)

dx_node, dw_node, dy_hat_node = tf.gradients(mse_node, [input_node, w_node, y_hat_node])
dx_tf = sess.run(dx_node)
dw_tf = sess.run(dw_node)
dy_hat_tf = sess.run(dy_hat_node)

tensors_saver.add(dx_tf.astype(np.float32))
tensors_saver.add(dw_tf.astype(np.float32))
tensors_saver.add(dy_hat_tf.astype(np.float32))
tensors_saver.add(y_hat_tf.astype(np.float32))
