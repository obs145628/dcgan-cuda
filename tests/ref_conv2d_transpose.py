import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

input = tf.constant([[
    [[-0.125],
     [-2.25]],
    [[-1.75],
     [-0.25]]],

    [[[-0.5],
      [3.0]],
     [[0.25],
     [-0.375]]]], dtype=tf.float32, name='input')

kernel = tf.constant([[
    [[1.0]],
    [[-1.0]]],
    [[[-1.0]],
    [[1.0]]]], dtype=tf.float32, name='kernel')

res = tf.nn.conv2d_transpose(input, kernel, [2, 4, 4, 1], [1, 2, 2, 1], "VALID")
sess = tf.Session()
conv = sess.run(res)
tensors_saver.add(conv)
