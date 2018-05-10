import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def main():
  # Import data
  mnist = input_data.read_data_sets(one_hot=True, train_dir='/tmp/tf/')
  

  # Create the model
  X = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])

  l1 = tf.layers.dense(X, units=100, activation=tf.nn.relu)
  l2 = tf.layers.dense(l1, units=10)
  
  # Define loss and optimizer
  


  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=l2))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  y_hat = tf.nn.softmax(l2)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    loss, _ = sess.run([cross_entropy, train_step], feed_dict={X: batch_xs, y: batch_ys})

    print('train loss = {}'.format(loss)) 
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('step {}: {}%'.format(i, sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      y: mnist.test.labels})))

if __name__ == '__main__':
  main()


  
