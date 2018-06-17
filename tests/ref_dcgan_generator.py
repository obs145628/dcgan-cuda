import os
import sys

import numpy as np
import tensorflow as tf
import gen_mnist
import tensors_saver

BATCH = 64
Z_DIM = 100

WEIGHTS_PATH = sys.argv[1]
tensors_saver.set_out_path(sys.argv[2])
new_weights = tensors_saver.Saver(WEIGHTS_PATH)

DATA_PATH = sys.argv[3]
celeba = np.load(DATA_PATH)
imgs = celeba['obj_000000']


imgs = np.ravel(imgs)
imgs = imgs[:BATCH*Z_DIM]
imgs = imgs.reshape(BATCH, Z_DIM)

X = tf.placeholder(tf.float32, shape=[None, Z_DIM])

l0 = tf.layers.dense(X, 512 * 4 * 4, activation=tf.nn.relu, name='dense0')
l0 = tf.reshape(l0, shape=[-1, 4, 4, 512])
w0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'dense0/kernel')[0]
b0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'dense0/bias')[0]

l1 = tf.layers.conv2d_transpose(l0, 256, (5, 5), strides=(2, 2),
                                padding='SAME', activation=tf.nn.relu, name='conv1')
w1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
b1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/bias')[0]

l2 = tf.layers.conv2d_transpose(l1, 128, (5, 5), strides=(2, 2),
                                padding='SAME', activation=tf.nn.relu, name='conv2')
w2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2/kernel')[0]
b2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2/bias')[0]

l3 = tf.layers.conv2d_transpose(l2, 64, (5, 5), strides=(2, 2),
                                padding='SAME', activation=tf.nn.relu, name='conv3')
w3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3/kernel')[0]
b3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3/bias')[0]

l4 = tf.layers.conv2d_transpose(l3, 3, (5, 5), strides=(2, 2),
                                padding='SAME', activation=None, name='conv4')
w4 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4/kernel')[0]
b4 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv4/bias')[0]


logits = l4
#prob = tf.nn.sigmoid(l4)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                              labels=tf.ones_like(logits)))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

l0_tf, l1_tf, l3_tf, logits_tf, loss_tf = sess.run([l0, l1, l3, logits, loss], feed_dict={X: imgs})

tensors_saver.add(l0_tf)
tensors_saver.add(l1_tf)
tensors_saver.add(l3_tf)
tensors_saver.add(logits_tf)
tensors_saver.add(loss_tf)


dw0 = tf.gradients(loss, w0)[0]
db0 = tf.gradients(loss, b0)[0]
dw1 = tf.gradients(loss, w1)[0]
db1 = tf.gradients(loss, b1)[0]
dw2 = tf.gradients(loss, w2)[0]
db2 = tf.gradients(loss, b2)[0]
dw3 = tf.gradients(loss, w3)[0]
db3 = tf.gradients(loss, b3)[0]
dw4 = tf.gradients(loss, w4)[0]
db4 = tf.gradients(loss, b4)[0]

tensors_saver.add(sess.run(dw0, feed_dict={X:imgs}))
tensors_saver.add(sess.run(db0, feed_dict={X:imgs}))
tensors_saver.add(sess.run(dw1, feed_dict={X:imgs}))
tensors_saver.add(sess.run(db1, feed_dict={X:imgs}))
tensors_saver.add(sess.run(dw2, feed_dict={X:imgs}))
tensors_saver.add(sess.run(db2, feed_dict={X:imgs}))
tensors_saver.add(sess.run(dw3, feed_dict={X:imgs}))
tensors_saver.add(sess.run(db3, feed_dict={X:imgs}))
tensors_saver.add(sess.run(dw4, feed_dict={X:imgs}))
tensors_saver.add(sess.run(db4, feed_dict={X:imgs}))

new_weights.add(sess.run(w0))
new_weights.add(sess.run(b0))
new_weights.add(sess.run(w1))
new_weights.add(sess.run(b1))
new_weights.add(sess.run(w2))
new_weights.add(sess.run(b2))
new_weights.add(sess.run(w3))
new_weights.add(sess.run(b3))
new_weights.add(sess.run(w4))
new_weights.add(sess.run(b4))

new_weights.save()

print(l0.get_shape())
print(l1.get_shape())
print(l2.get_shape())
print(l3.get_shape())
print(l4.get_shape())

print(w0.get_shape())
print(b0.get_shape())
print(w1.get_shape())
print(b1.get_shape())
print(w2.get_shape())
print(b2.get_shape())
print(w3.get_shape())
print(b3.get_shape())
print(w4.get_shape())
print(b4.get_shape())
