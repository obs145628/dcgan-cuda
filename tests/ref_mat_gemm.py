import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

tensors_saver.set_out_path(sys.argv[1])

a = np.random.random_integers(-50, 50, size=(512,512))
b = np.random.random_integers(-50, 50, size=(512,512))

a_node = tf.Variable(a, dtype=tf.float32);
b_node = tf.Variable(b, dtype=tf.float32);
res_node = tf.matmul(a_node, b_node);
sess = tf.Session();
init = tf.global_variables_initializer();
sess.run(init);

tf_res = sess.run(res_node);
tensors_saver.add(a.astype(np.float32));
tensors_saver.add(b.astype(np.float32));
tensors_saver.save()
tensors_saver.clear()
tensors_saver.set_out_path(sys.argv[2])
tensors_saver.add(tf_res.astype(np.float32));
