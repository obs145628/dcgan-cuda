import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver
import xorshift

from tensorflow.python.client import timeline

tensors_saver.set_out_path(sys.argv[1])
xorshift.seed(234)

BATCH = 10

x = xorshift.np_f32((BATCH, 64, 64, 3))
w = xorshift.np_f32((5, 5, 3, 64))

x_node = tf.Variable(x)
w_node = tf.Variable(w)
y_node = tf.nn.conv2d(x_node, w_node, [1, 2, 2, 1], 'SAME')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metada = tf.RunMetadata()
y = sess.run(y_node, options=run_options, run_metadata=run_metada)
tensors_saver.add(y)

ctf = timeline.Timeline(run_metada.step_stats).generate_chrome_trace_format()
with open('timeline.json', 'w') as f:
    f.write(ctf)
