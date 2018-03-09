# Day_01_04_softmax_multilayer.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist')

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32)
keep_rate = tf.placeholder(tf.float32)

# ---------------------------------- #

w1 = tf.get_variable('w1', shape=[784, 256],
        initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable('w2', shape=[256, 256],
        initializer=tf.contrib.layers.xavier_initializer())
w3 = tf.get_variable('w3', shape=[256, 10],
        initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([10]))

z1 = tf.matmul(x, w1) + b1
r1 = tf.nn.relu(z1)
d1 = tf.nn.dropout(r1, keep_rate)

z2 = tf.matmul(d1, w2) + b2
r2 = tf.nn.relu(z2)
d2 = tf.nn.dropout(r2, keep_rate)

z = tf.matmul(d2, w3) + b3
hypothesis = tf.nn.softmax(z)
cost_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=z)
cost = tf.reduce_mean(cost_i)
train = tf.train.AdamOptimizer(0.001).minimize(cost)

# ---------------------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs, batch_size = 30, 100
count = mnist.train.num_examples // batch_size
for i in range(epochs):
    for _ in range(count):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        y_batch = np.int32(y_batch)
        sess.run(train, feed_dict={x: x_batch,
                                   y: y_batch,
                                   keep_rate: 0.7})
    print('epochs :', i)

target_test = np.int32(mnist.test.labels)

prediction = tf.equal(tf.argmax(hypothesis, 1), target_test)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print(sess.run(accuracy,
               feed_dict={x: mnist.test.images,
                          y: target_test,
                          keep_rate: 1.0}))

sess.close()

# 10 : 0.9791
# 30 : 0.9825
