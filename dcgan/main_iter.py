import os
import glob
import sys
import utils
import random
import time

import numpy as np
import tensorflow as tf
import gen_mnist
import tensors_saver

np.random.seed(12)
tf.set_random_seed(12)

EPOCHS = 25 #number of training epochs
LEARNING_RATE = 0.0002 #adam optimizer learning rate
BETA1 = 0.5 #adam optimizer beta1 parameter
BATCH_SIZE = 64 #number of inputs per batch
Z_DIM = 100 #size of the noise input vector to generate image

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
X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

g_logits, g_out, g_weights, g_nodes = generator(z, reuse=False)
d1_logits, d1_out, d1_weights, d1_nodes = discriminator(g_out, reuse=False)
d2_logits, d2_out, d2_weights, d2_nodes = discriminator(X, reuse=True)

g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(d1_logits),
                                                logits=d1_logits))
d1_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(d1_logits),
                                                logits=d1_logits))

d2_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(d2_logits),
                                                logits=d2_logits))
d_loss = d1_loss + d2_loss

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

# optimizers for updating discriminator and generator
d_opti = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(d_loss, var_list=d_vars)
g_opti = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(g_loss, var_list=g_vars)

#d_opti = tf.train.GradientDescentOptimizer(0.5).minimize(d_loss, var_list=d_vars)
#g_opti = tf.train.GradientDescentOptimizer(0.5).minimize(g_loss, var_list=g_vars)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

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
d_dw0 = tf.gradients(d_loss, d1_weights[0])[0]
d_db0 = tf.gradients(d_loss, d1_weights[1])[0]
d_dw1 = tf.gradients(d_loss, d1_weights[2])[0]
d_db1 = tf.gradients(d_loss, d1_weights[3])[0]
d_dw2 = tf.gradients(d_loss, d1_weights[4])[0]
d_db2 = tf.gradients(d_loss, d1_weights[5])[0]
d_dw3 = tf.gradients(d_loss, d1_weights[6])[0]
d_db3 = tf.gradients(d_loss, d1_weights[7])[0]
d_dw4 = tf.gradients(d_loss, d1_weights[8])[0]
d_db4 = tf.gradients(d_loss, d1_weights[9])[0]

data_files = glob.glob(os.path.join("../celeba_norm", "*.jpg"))
data_x = data_files[:BATCH_SIZE]
data_x = [utils.get_image(f) for f in data_x]
data_x = np.array(data_x).astype(np.float32)
data_z = np.random.normal(loc=0.0, scale=1.0,
                          size=(BATCH_SIZE, Z_DIM)).astype(np.float32)

inputt = tensors_saver.Saver('input.npz')
inputt.add(data_x)
inputt.add(data_z)
inputt.save()

def save_grads(path):
    gw0, gb0, gw1, gb1, gw2, gb2, gw3, gb3, gw4, gb4, dw0, db0, dw1, db1, dw2, db2, dw3, db3, dw4, db4 = sess.run([
        g_dw0, g_db0, g_dw1, g_db1, g_dw2, g_db2, g_dw3, g_db3, g_dw4, g_db4,
        d_dw0, d_db0, d_dw1, d_db1, d_dw2, d_db2, d_dw3, d_db3, d_dw4, d_db4],
                                                                feed_dict={X: data_x, z: data_z})
    wi = tensors_saver.Saver(path)
    wi.add(gw0)
    wi.add(gb0)
    wi.add(gw1)
    wi.add(gb1)
    wi.add(gw2)
    wi.add(gb2)
    wi.add(gw3)
    wi.add(gb3)
    wi.add(gw4)
    wi.add(gb4)
    wi.add(dw0)
    wi.add(db0)
    wi.add(dw1)
    wi.add(db1)
    wi.add(dw2)
    wi.add(db2)
    wi.add(dw3)
    wi.add(db3)
    wi.add(dw4)
    wi.add(db4)
    wi.save()

def save_weights(path):
    wi = tensors_saver.Saver(path)
    wi.add(sess.run(g_weights[0]))
    wi.add(sess.run(g_weights[1]))
    wi.add(sess.run(g_weights[2]))
    wi.add(sess.run(g_weights[3]))
    wi.add(sess.run(g_weights[4]))
    wi.add(sess.run(g_weights[5]))
    wi.add(sess.run(g_weights[6]))
    wi.add(sess.run(g_weights[7]))
    wi.add(sess.run(g_weights[8]))
    wi.add(sess.run(g_weights[9]))
    wi.add(sess.run(d1_weights[0]))
    wi.add(sess.run(d1_weights[1]))
    wi.add(sess.run(d1_weights[2]))
    wi.add(sess.run(d1_weights[3]))
    wi.add(sess.run(d1_weights[4]))
    wi.add(sess.run(d1_weights[5]))
    wi.add(sess.run(d1_weights[6]))
    wi.add(sess.run(d1_weights[7]))
    wi.add(sess.run(d1_weights[8]))
    wi.add(sess.run(d1_weights[9]))
    wi.save()


for i in range(3000):

    print('iteration ' + str(i))
    img = sess.run(g_out, feed_dict={z : data_z})
    utils.save_images(img, [8, 8], 'sample' + str(i) + '.png')
    save_grads('grads' + str(i) + '.npz')
    save_weights('weights' + str(i) + '.npz')

    sess.run(d_opti, feed_dict={z: data_z, X: data_x})
    for _ in range(2):
        sess.run(g_opti, feed_dict={z: data_z})

'''
# Training 
i = 0
for epoch in range(EPOCHS):
    random.shuffle(data_files)
    batch_idxs = len(data_files) // BATCH_SIZE

    for idx in range(0, batch_idxs):
        batch_files = data_files[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        batch = [utils.get_image(f) for f in batch_files]
        batch_images = np.array(batch).astype(np.float32)
        batch_z = np.random.normal(loc=0.0, scale=1.0,
                                       size=(SAMPLE_SIZE, Z_DIM)).astype(np.float32)
        start_time = time.time()

        # updates the discriminator
        errD, _ = sess.run([d_loss, d_opti], feed_dict={z: batch_z, X: batch_images })
        for _ in range(2):
            errG, _ = sess.run([g_loss, g_opti], feed_dict={z: batch_z})
        print("Epoch %2d: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, idx, batch_idxs, time.time() - start_time, errD, errG))

        i += 1
        if i % SAMPLE_STEP == 0:
            img = sess.run(g_out, feed_dict={z : sample_seed})
            utils.save_images(img, [8, 8],
                              './{}/train_{:02d}_{:04d}.png'.format(SAMPLE_DIR, epoch, idx))
            save_net('./{}/train_{:02d}_{:04d}.npz'.format(SAMPLE_DIR, epoch, idx))
            print('sample generated')
'''
