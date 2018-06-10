import glob
import os
import random
import time

import numpy as np
import tensorflow as tf

import model
import utils




EPOCHS = 25 #number of training epochs
LEARNING_RATE = 0.0002 #adam optimizer learning rate
BETA1 = 0.5 #adam optimizer beta1 parameter
BATCH_SIZE = 64 #number of inputs per batch
IMAGE_SIZE = 108 #image size (before center / crop)
OUTPUT_SIZE = 64 #size of generated images
SAMPLE_SIZE = 64 #number of samples generated
SAMPLE_STEP = 5 #interval of steps between each sample generation
#SAVE_STEP = 500 #interval of steps between each saving of the network
SAMPLE_DIR = 'samples' #folder where the samples are saved
Z_DIM = 100 #size of the noise input vector to generate image


def main():
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    z = tf.placeholder(tf.float32, [BATCH_SIZE, Z_DIM])
    X = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3])

    # generator
    g_logits, g_out, g_weights = model.generator(z, reuse=False)
    # discriminator of fake images
    d1_logits, d1_out, d1_weights = model.discriminator(g_out, reuse=False)
    # discriminator of real images
    d2_logits, d2_out, d2_weights = model.discriminator(X, reuse=True)


    # Losses 
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

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    data_files = glob.glob(os.path.join("../celebA", "*.jpg"))
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(SAMPLE_SIZE, Z_DIM)).astype(np.float32)



    # Training 
    i = 0
    for epoch in range(EPOCHS):
        random.shuffle(data_files)
        batch_idxs = len(data_files) // BATCH_SIZE

        for idx in range(0, batch_idxs):
            batch_files = data_files[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch = [utils.get_image(f, IMAGE_SIZE, OUTPUT_SIZE) for f in batch_files]
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
                print('sample generated')

if __name__ == '__main__':
    main()
