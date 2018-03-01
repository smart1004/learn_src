# Day_02_06_tflayers_mnist_1.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

# ---------------------------------- #

conv1 = tf.layers.conv2d(x, 32, [3, 3], activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

conv2 = tf.layers.conv2d(pool1, 64, [3, 3], activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])

flats = tf.layers.flatten(pool2)

fc1 = tf.layers.dense(flats, 256, activation=tf.nn.relu)
drops = tf.layers.dropout(fc1, 0.5, is_training)

hypothesis = tf.layers.dense(drops, 10)

# -------------------------- #

cost_i = tf.nn.softmax_cross_entropy_with_logits(
      labels=y, logits=hypothesis)
cost = tf.reduce_mean(cost_i)
train = tf.train.AdamOptimizer(0.001).minimize(cost)

# ---------------------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs, batch_size = 1, 100
count = mnist.train.num_examples // batch_size
for i in range(epochs):
    for _ in range(count):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch = x_batch.reshape(-1, 28, 28, 1)

        sess.run(train, feed_dict={x: x_batch,
                                   y: y_batch,
                                   is_training: True})
    print('epochs :', i)

prediction = tf.equal(tf.argmax(hypothesis, 1),
                      tf.argmax(mnist.test.labels, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print(sess.run(accuracy,
               feed_dict={x: mnist.test.images.reshape(-1, 28, 28, 1),
                          is_training: False}))

sess.close()
