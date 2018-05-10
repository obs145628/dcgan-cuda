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

def find_weight(x):
    return tf.get_default_graph().get_tensor_by_name(
        os.path.split(x.name)[0] + '/kernel:0')

def find_bias(x):
    return tf.get_default_graph().get_tensor_by_name(
        os.path.split(x.name)[0] + '/bias:0')

X = tf.placeholder(tf.float32, (None, 784))
y = tf.placeholder(tf.float32, (None, 10))

X_train, y_train = gen_mnist.get_dataset()



l1 = layer_dense(X, 100, tf.sigmoid)
w1 = find_weight(l1)
b1 = find_bias(l1)

l2 = layer_dense(l1, 10)
w2 = find_weight(l2)
b2 = find_bias(l2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=l2))

dx, dw1, db1 = tf.gradients(loss, [X, w1, b1])
dl1, dw2, db2 = tf.gradients(loss, [l1, w2, b2])
dl2 = tf.gradients(loss, [l2])[0]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


data = {X: X_train[0:20], y: y_train[0:20]};
data2 = {X: X_train[20:50], y: y_train[20:50]};
data3 = {X: X_train[50:80], y: y_train[50:80]};

tensors_saver.add(sess.run(l2, feed_dict=data))
tensors_saver.add(sess.run(loss, feed_dict=data))
tensors_saver.add(sess.run(dx, feed_dict=data))
tensors_saver.add(sess.run(dw1, feed_dict=data))
tensors_saver.add(sess.run(db1, feed_dict=data))
tensors_saver.add(sess.run(dl1, feed_dict=data))
tensors_saver.add(sess.run(dw2, feed_dict=data))
tensors_saver.add(sess.run(db2, feed_dict=data))
tensors_saver.add(sess.run(dl2, feed_dict=data))


tensors_saver.add(sess.run(l2, feed_dict=data2))
tensors_saver.add(sess.run(loss, feed_dict=data2))
tensors_saver.add(sess.run(dx, feed_dict=data2))
tensors_saver.add(sess.run(dw1, feed_dict=data2))
tensors_saver.add(sess.run(db1, feed_dict=data2))
tensors_saver.add(sess.run(dl1, feed_dict=data2))
tensors_saver.add(sess.run(dw2, feed_dict=data2))
tensors_saver.add(sess.run(db2, feed_dict=data2))
tensors_saver.add(sess.run(dl2, feed_dict=data2))

tensors_saver.add(sess.run(l2, feed_dict=data3))
tensors_saver.add(sess.run(loss, feed_dict=data3))
tensors_saver.add(sess.run(dx, feed_dict=data3))
tensors_saver.add(sess.run(dw1, feed_dict=data3))
tensors_saver.add(sess.run(db1, feed_dict=data3))
tensors_saver.add(sess.run(dl1, feed_dict=data3))
tensors_saver.add(sess.run(dw2, feed_dict=data3))
tensors_saver.add(sess.run(db2, feed_dict=data3))
tensors_saver.add(sess.run(dl2, feed_dict=data3))

if weights is None:
    new_weights.save()


