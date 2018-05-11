import numpy as np
import tensorflow as tf
from PIL import Image

path = "../models/dcgan-tf/data/celebA/035749.jpg"


im = Image.open(path)
im = im.resize((64, 64), Image.ANTIALIAS)
im = np.array(im)
imgs = im.reshape(1, 64, 64, 3)
print(imgs.shape)

# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x):

    l0 = tf.layers.conv2d(x, 64, (5, 5), (2, 2), activation=tf.nn.leaky_relu,
                          padding='SAME')

    l1 = tf.layers.conv2d(l0, 128, (5, 5), (2, 2), activation=None,
                          padding='SAME')

    l2 = tf.layers.conv2d(l1, 256, (5, 5), (2, 2), activation=None,
                          padding='SAME')

    l3 = tf.layers.conv2d(l2, 512, (5, 5), (2, 2), activation=None,
                          padding='SAME')

    l4 = tf.layers.flatten(l3)
    l4 = tf.layers.dense(l4, 1, activation=None)
    return l4, tf.nn.sigmoid(l4)

X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    
logits, prob = discriminator(X)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                              labels=tf.ones_like(logits)))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)




logits_tf, loss_tf = sess.run([logits, loss], feed_dict={X: imgs})

print(logits_tf)
print(loss_tf)

