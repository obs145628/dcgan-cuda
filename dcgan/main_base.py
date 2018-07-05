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

def generator(X, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope('generator', reuse=reuse):
        l0 = tf.layers.dense(X, 512 * 4 * 4, activation=tf.nn.relu, name='dense0',
                             kernel_initializer=w_init)
        l0 = tf.reshape(l0, shape=[-1, 4, 4, 512])
        w0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/dense0/kernel')[0]
        b0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/dense0/bias')[0]

        l1 = tf.layers.conv2d_transpose(l0, 256, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=tf.nn.relu, name='conv1',
                                        kernel_initializer=w_init)
        w1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv1/kernel')[0]
        b1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv1/bias')[0]

        l2 = tf.layers.conv2d_transpose(l1, 128, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=tf.nn.relu, name='conv2',
                                        kernel_initializer=w_init)
        w2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv2/kernel')[0]
        b2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv2/bias')[0]

        l3 = tf.layers.conv2d_transpose(l2, 64, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=tf.nn.relu, name='conv3',
                                        kernel_initializer=w_init)
        w3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv3/kernel')[0]
        b3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv3/bias')[0]

        l4 = tf.layers.conv2d_transpose(l3, 3, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=None, name='conv4',
                                        kernel_initializer=w_init)
        w4 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv4/kernel')[0]
        b4 = tf.get_collection(tf.GraphKeys.VARIABLES, 'generator/conv4/bias')[0]

    logits = l4
    output = tf.nn.tanh(logits)
    weights = [w0, b0, w1, b1, w2, b2, w3, b3, w4, b4]
    nodes = [l0, l1, l2, l3, l4]
    return logits, output, weights, nodes

def discriminator(X, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope('discriminator', reuse=reuse):
    
        l0 = tf.layers.conv2d(X, 64, (5, 5), (2, 2), activation=tf.nn.leaky_relu,
                              padding='SAME', name='conv0', kernel_initializer=w_init)
        w0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv0/kernel')[0]
        b0 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv0/bias')[0]


        l1 = tf.layers.conv2d(l0, 128, (5, 5), (2, 2), activation=tf.nn.leaky_relu,
                              padding='SAME', name='conv1', kernel_initializer=w_init)
        w1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv1/kernel')[0]
        b1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv1/bias')[0]
    
        l2 = tf.layers.conv2d(l1, 256, (5, 5), (2, 2), activation=tf.nn.leaky_relu,
                              padding='SAME', name='conv2', kernel_initializer=w_init)
        w2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv2/kernel')[0]
        b2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv2/bias')[0]

        l3 = tf.layers.conv2d(l2, 512, (5, 5), (2, 2), activation=tf.nn.leaky_relu,
                              padding='SAME', name='conv3', kernel_initializer=w_init)
        w3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv3/kernel')[0]
        b3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/conv3/bias')[0]

        l4 = tf.layers.flatten(l3)
        l4 = tf.layers.dense(l4, 1, activation=None, name='dense', kernel_initializer=w_init)
        w4 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/dense/kernel')[0]
        b4 = tf.get_collection(tf.GraphKeys.VARIABLES, 'discriminator/dense/bias')[0]

    logits = l4
    output = tf.nn.sigmoid(l4)
    weights = [w0, b0, w1, b1, w2, b2, w3, b3, w4, b4]
    nodes = [l0, l1, l2, l3, l4]
    return logits, output, weights, nodes

z = tf.placeholder(tf.float32, shape=[None, Z_DIM])

g_logits, g_out, g_weights, g_nodes = generator(z, reuse=False)
d1_logits, d1_out, d1_weights, d1_nodes = discriminator(g_out, reuse=False)

g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(d1_logits),
                                                logits=d1_logits))
d1_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(d1_logits),
                                                logits=d1_logits))



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

l0_tf, l1_tf, l2_tf, l3_tf, logits_tf, out_tf, loss_tf = sess.run([g_nodes[0], g_nodes[1], g_nodes[2], g_nodes[3], g_logits, g_out, g_loss], feed_dict={z: imgs})

tensors_saver.add(l0_tf)
tensors_saver.add(l1_tf)
tensors_saver.add(l2_tf)
tensors_saver.add(l3_tf)
tensors_saver.add(logits_tf)
tensors_saver.add(out_tf)

tensors_saver.add(sess.run(d1_nodes[0], feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_nodes[1], feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_nodes[2], feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_nodes[3], feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_logits, feed_dict={z:imgs}))

tensors_saver.add(loss_tf)


g_dw0 = tf.gradients(g_loss, g_weights[0])[0]
g_db0 = tf.gradients(g_loss, g_weights[1])[0]
g_dw1 = tf.gradients(g_loss, g_weights[2])[0]
g_db1 = tf.gradients(g_loss, g_weights[3])[0]
g_dw2 = tf.gradients(g_loss, g_weights[4])[0]
g_db2 = tf.gradients(g_loss, g_weights[5])[0]
g_dw3 = tf.gradients(g_loss, g_weights[6])[0]
g_db3 = tf.gradients(g_loss, g_weights[7])[0]
g_dw4 = tf.gradients(g_loss, g_weights[8])[0]
g_db4 = tf.gradients(g_loss, g_weights[9])[0]

d1_dw0 = tf.gradients(d1_loss, d1_weights[0])[0]
d1_db0 = tf.gradients(d1_loss, d1_weights[1])[0]
d1_dw1 = tf.gradients(d1_loss, d1_weights[2])[0]
d1_db1 = tf.gradients(d1_loss, d1_weights[3])[0]
d1_dw2 = tf.gradients(d1_loss, d1_weights[4])[0]
d1_db2 = tf.gradients(d1_loss, d1_weights[5])[0]
d1_dw3 = tf.gradients(d1_loss, d1_weights[6])[0]
d1_db3 = tf.gradients(d1_loss, d1_weights[7])[0]
d1_dw4 = tf.gradients(d1_loss, d1_weights[8])[0]
d1_db4 = tf.gradients(d1_loss, d1_weights[9])[0]

tensors_saver.add(sess.run(g_dw0, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_db0, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_dw1, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_db1, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_dw2, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_db2, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_dw3, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_db3, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_dw4, feed_dict={z:imgs}))
tensors_saver.add(sess.run(g_db4, feed_dict={z:imgs}))

tensors_saver.add(sess.run(d1_dw0, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_db0, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_dw1, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_db1, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_dw2, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_db2, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_dw3, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_db3, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_dw4, feed_dict={z:imgs}))
tensors_saver.add(sess.run(d1_db4, feed_dict={z:imgs}))

new_weights.add(sess.run(g_weights[0]))
new_weights.add(sess.run(g_weights[1]))
new_weights.add(sess.run(g_weights[2]))
new_weights.add(sess.run(g_weights[3]))
new_weights.add(sess.run(g_weights[4]))
new_weights.add(sess.run(g_weights[5]))
new_weights.add(sess.run(g_weights[6]))
new_weights.add(sess.run(g_weights[7]))
new_weights.add(sess.run(g_weights[8]))
new_weights.add(sess.run(g_weights[9]))
new_weights.add(sess.run(d1_weights[0]))
new_weights.add(sess.run(d1_weights[1]))
new_weights.add(sess.run(d1_weights[2]))
new_weights.add(sess.run(d1_weights[3]))
new_weights.add(sess.run(d1_weights[4]))
new_weights.add(sess.run(d1_weights[5]))
new_weights.add(sess.run(d1_weights[6]))
new_weights.add(sess.run(d1_weights[7]))
new_weights.add(sess.run(d1_weights[8]))
new_weights.add(sess.run(d1_weights[9]))
new_weights.save()
