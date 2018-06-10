import tensorflow as tf

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
    return logits, output, weights

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
    return logits, output, weights
