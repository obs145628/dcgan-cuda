import os
import sys

import numpy as np
import tensorflow as tf
import gen_mnist
import tensors_saver

WEIGHTS_PATH = sys.argv[1]
tensors_saver.set_out_path(sys.argv[2])

def find_weight(x):
    return tf.get_default_graph().get_tensor_by_name(
        os.path.split(x.name)[0] + '/kernel:0')

def find_bias(x):
    return tf.get_default_graph().get_tensor_by_name(
        os.path.split(x.name)[0] + '/bias:0')

X = tf.placeholder(tf.float32, (None, 784))
y = tf.placeholder(tf.float32, (None, 10))

X_train, y_train = gen_mnist.get_dataset()


l1 = tf.layers.dense(X, 100, activation= tf.nn.relu)
w1 = find_weight(l1)
b1 = find_bias(l1)

l2 = tf.layers.dense(l1, 10)
w2 = find_weight(l2)
b2 = find_bias(l2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=l2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

data = {X: X_train[0:20], y: y_train[0:20]};


if os.path.exists(WEIGHTS_PATH):
    print('get old weights')
    weights = [x for _, x in np.load(WEIGHTS_PATH).items()]
    sess.run(tf.assign(w1, weights[0]))
    sess.run(tf.assign(b1, weights[1]))
    sess.run(tf.assign(w2, weights[2]))
    sess.run(tf.assign(b2, weights[3]))
else:
    print('save new weights')
    weights = tensors_saver.Saver(WEIGHTS_PATH)
    weights.add(sess.run(w1))
    weights.add(sess.run(b1))
    weights.add(sess.run(w2))
    weights.add(sess.run(b2))
    weights.save()


sess.run(train_op, feed_dict=data)

tensors_saver.add(sess.run(w1))
tensors_saver.add(sess.run(b1))
tensors_saver.add(sess.run(w2))
tensors_saver.add(sess.run(b2))

print(sess.run(loss, feed_dict=data))

#print(w1.get_shape())
#print(b1.get_shape())
#print(w2.get_shape())
#print(b2.get_shape())

