import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

input = tf.constant([[
    [
        [5, 0],
        [-1, -1],
        [2, -2],
        [3, 4]
    ],[
        [3, 3],
        [6, 2],
        [3, 4],
        [4, 0]
    ],[
        [1, 0],
        [0, -2],
        [0, 1],
        [2, 0]
    ],[
        [3, 6],
        [-2, 7],
        [-2, 1],
        [1, 0]
    ]
   ],[
    [
        [8, 2],
        [0, 1],
        [4, -2],
        [1, -1]
    ],[
        [3, 0],
        [4, 2],
        [3, 0],
        [-3, 0]
    ],[
        [1, 0],
        [-8, 7],
        [0, 1],
        [0, -9]
    ],[
        [3, 0],
        [2, -7],
        [-2, 1],
        [-1, 0]
    ]
   ]
], dtype=tf.float32, name='input')

kernel = tf.constant([
    [
        [
            [1, -1],
            [-1, 0]
        ],
        [
            [0, 0],
            [-1, 1]
        ]
    ],[
        [
            [0, -1],
            [0, -1]
        ],
        [
            [1, 1],
            [-1, 0]
        ]
    ]
], dtype=tf.float32, name='kernel')

res = tf.squeeze(tf.nn.conv2d(input, kernel, [1, 2, 2, 1], "VALID"))
sess = tf.Session()
conv = sess.run(res)
tensors_saver.add(conv)
