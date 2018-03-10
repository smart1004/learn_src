# Day_01_05_cnn_basic.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist')

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32)
keep_rate = tf.placeholder(tf.float32)

# ---------------------------------- #

w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
w_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 256], stddev=0.1))
w_fc2 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))

b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

# ---------------------------------- #

x_image = tf.reshape(x, [-1, 28, 28, 1])

# (?, 28, 28, 32)
c_conv1 = tf.nn.conv2d(x_image, w_conv1,
                       strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(c_conv1 + b_conv1)
# (?, 14, 14, 32)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

# (?, 14, 14, 64)
c_conv2 = tf.nn.conv2d(h_pool1, w_conv2,
                       strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(c_conv2 + b_conv2)
# (?, 7, 7, 64)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

# 컨볼루션 --> 풀리 커넥티드
h_flats = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_flats, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_rate)

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
# hypothesis = tf.nn.softmax(y_conv)
hypothesis = y_conv

# -------------------------- #

cost_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=y_conv)
cost = tf.reduce_mean(cost_i)
train = tf.train.AdamOptimizer(0.001).minimize(cost)

# ---------------------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs, batch_size = 15, 100
count = mnist.train.num_examples // batch_size
for i in range(epochs):
    for _ in range(count):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        y_batch = np.int32(y_batch)
        sess.run(train, feed_dict={x: x_batch,
                                   y: y_batch,
                                   keep_rate: 0.5})
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
