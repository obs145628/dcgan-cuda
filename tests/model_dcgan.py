import tensorflow as tf

def batch_norm_layer(x, gamma_init, is_train, activation):
       y = tf.layers.batch_normalization(x, training=is_train,
                                             gamma_initializer=gamma_init)
       #y = activation(y)
       return y

# Generator Network
# Input: Noise, Output: Image
def generator(x, is_train, reuse):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    
    with tf.variable_scope('generator', reuse=reuse):

        l0 = tf.layers.dense(x, units=512*4*4, kernel_initializer=w_init)
        l0 = tf.reshape(l0, shape=[-1, 4, 4, 512])
        l0 = batch_norm_layer(l0, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.relu)
        
        l1 = tf.layers.conv2d_transpose(l0, 256, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=None,
                                        kernel_initializer=w_init)
        l1 = batch_norm_layer(l1, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.relu)

        l2 = tf.layers.conv2d_transpose(l1, 128, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=None,
                                        kernel_initializer=w_init)
        l2 = batch_norm_layer(l2, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.relu)

        l3 = tf.layers.conv2d_transpose(l2, 64, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=None,
                                        kernel_initializer=w_init)
        l3 = batch_norm_layer(l3, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.relu)

        l4 = tf.layers.conv2d_transpose(l3, 3, (5, 5), strides=(2, 2),
                                        padding='SAME', activation=None,
                                        kernel_initializer=w_init)
        l4 = batch_norm_layer(l4, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.relu)

        return l4, tf.nn.tanh(l4)


# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, is_train, reuse):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope('discriminator', reuse=reuse):

        l0 = tf.layers.conv2d(x, 64, (5, 5), (2, 2), activation=tf.nn.leaky_relu,
                              padding='SAME', kernel_initializer=w_init)

        l1 = tf.layers.conv2d(l0, 128, (5, 5), (2, 2), activation=None,
                              padding='SAME', kernel_initializer=w_init)
        l1 = batch_norm_layer(l1, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.leaky_relu)

        l2 = tf.layers.conv2d(l1, 256, (5, 5), (2, 2), activation=None,
                              padding='SAME', kernel_initializer=w_init)
        l2 = batch_norm_layer(l2, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.leaky_relu)

        l3 = tf.layers.conv2d(l2, 512, (5, 5), (2, 2), activation=None,
                              padding='SAME', kernel_initializer=w_init)
        l3 = batch_norm_layer(l3, gamma_init=gamma_init, is_train=is_train, activation=tf.nn.leaky_relu)

        l4 = tf.layers.flatten(l3)
        l4 = tf.layers.dense(l4, 1, activation=None, kernel_initializer=w_init)
        return l4, tf.nn.sigmoid(l4)
