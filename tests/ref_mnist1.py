import os
import sys

import numpy as np
import tensorflow as tf
import gen_mnist
import tensors_saver

WEIGHTS_PATH = sys.argv[1]
tensors_saver.set_out_path(sys.argv[2])

weights = None
new_weights = tensors_saver.Saver(WEIGHTS_PATH)
if os.path.exists(WEIGHTS_PATH):
    weights = [x for _, x in np.load(WEIGHTS_PATH).items()]


class TInit:
    
    def __init__(self, x):
        self.x = x
        
    def __call__(self, shape, dtype, partition_info):
        return self.x
    


def layer_dense(input, noutputs, activation = None):
    if weights is not None:
        w = weights.pop(0)
        b = weights.pop(0)
    else:
        w = np.random.randn(input.shape[1], noutputs).astype(np.float32)
        b = np.random.randn(noutputs).astype(np.float32)
        new_weights.add(w)
        new_weights.add(b)

    return tf.layers.dense(input, 100,
                           activation=activation,
                           kernel_initializer=TInit(w),
                           bias_initializer=TInit(b)
    )

X = tf.placeholder(tf.float32, (None, 784))
y = tf.placeholder(tf.float32, (None, 10))

X_train, y_train = gen_mnist.get_dataset()


l1 = layer_dense(X, 100, tf.sigmoid)
l2 = layer_dense(l1, 10, tf.sigmoid)
loss = tf.losses.mean_squared_error(y, l2)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


data = {X: X_train[0:20], y: y_train[0:20]};
data2 = {X: X_train[20:50], y: y_train[20:50]};
data3 = {X: X_train[50:80], y: y_train[50:80]};

tensors_saver.add(sess.run(l2, feed_dict=data))
tensors_saver.add(sess.run(loss, feed_dict=data))

tensors_saver.add(sess.run(l2, feed_dict=data2))
tensors_saver.add(sess.run(loss, feed_dict=data2))

tensors_saver.add(sess.run(l2, feed_dict=data3))
tensors_saver.add(sess.run(loss, feed_dict=data3))


if weights is None:
    new_weights.save()
