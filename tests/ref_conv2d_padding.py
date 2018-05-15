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

res = tf.nn.conv2d(input, kernel, [1, 2, 2, 1], "SAME")
sess = tf.Session()
conv = sess.run(res)
tensors_saver.add(conv)
