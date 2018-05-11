


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets(one_hot=True, train_dir='/tmp/tf/')


X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

l0 = tf.reshape(X, shape=[-1, 28, 28, 1])
l1 = tf.layers.conv2d(
    inputs=l0,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
l1 = tf.layers.max_pooling2d(inputs=l1, pool_size=[2, 2], strides=2)

l2 = tf.layers.conv2d(
    inputs=l1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
l2 = tf.layers.max_pooling2d(inputs=l2, pool_size=[2, 2], strides=2)
l2 = tf.reshape(l2, shape=[-1, 7 * 7 * 64])

l3 = tf.layers.dense(inputs=l2, units=1024, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=l3, units=10)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    loss, _ = sess.run([loss_op, train_op], feed_dict={X: batch_xs, y: batch_ys})

    print('train loss = {}'.format(loss)) 
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('step {}: {}%'.format(i, sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      y: mnist.test.labels})))
