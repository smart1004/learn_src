# Day_03_06_auto_encoder.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def make_model_1():
    # encoder
    w1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.zeros([256]))

    # decoder
    w2 = tf.Variable(tf.random_normal([256, 784]))
    b2 = tf.Variable(tf.zeros([784]))

    # encoder
    z1 = tf.matmul(x, w1) + b1
    d1 = tf.nn.sigmoid(z1)

    # decoder
    z2 = tf.matmul(d1, w2) + b2
    return tf.nn.sigmoid(z2)


# 문제
# xavier 초기화
# 784 - 256 - 128 - 256 - 784
def make_model_2():
    # encoder
    w1 = tf.get_variable('w1', shape=[784, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([256]))

    w2 = tf.get_variable('w2', shape=[256, 128],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([128]))

    w3 = tf.get_variable('w3', shape=[128, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.zeros([256]))

    w4 = tf.get_variable('w4', shape=[256, 784],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.zeros([784]))

    # ------------------------------------ #

    z1 = tf.matmul(x, w1) + b1
    d1 = tf.nn.relu(z1)

    z2 = tf.matmul(d1, w2) + b2
    d2 = tf.nn.relu(z2)

    z3 = tf.matmul(d2, w3) + b3
    d3 = tf.nn.relu(z3)

    z4 = tf.matmul(d3, w4) + b4
    return tf.nn.sigmoid(z4)


def make_model_3():
    input = x
    dense1 = tf.layers.dense(inputs=input, units=256, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=16, activation=tf.nn.relu)
    dense4 = tf.layers.dense(inputs=dense3, units=128, activation=tf.nn.relu)
    dense5 = tf.layers.dense(inputs=dense4, units=256, activation=tf.nn.relu)
    return   tf.layers.dense(inputs=dense5, units=784, activation=tf.nn.sigmoid)


mnist = input_data.read_data_sets('mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

# output = make_model_1()
# output = make_model_2()
output = make_model_3()

cost = tf.reduce_mean((x - output) ** 2)
train = tf.train.RMSPropOptimizer(0.01).minimize(cost)

# ---------------------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs, batch_size = 15, 128

loops = mnist.train.num_examples // batch_size
print('loops :', loops)

for i in range(epochs):
    total = 0
    for _ in range(loops):
        xx, _ = mnist.train.next_batch(batch_size)

        _, loss = sess.run([train, cost], feed_dict={x: xx})
        total += loss

    print('{:2} : {}'.format(i, total / loops))

# ---------------------------------- #

sample_count = 10
samples = sess.run(output, {x: mnist.test.images[:sample_count]})

_, ax = plt.subplots(2, sample_count, figsize=[sample_count, 2])

for i in range(sample_count):
    ax[0, i].set_axis_off()
    ax[1, i].set_axis_off()
    ax[0, i].imshow(np.reshape(mnist.test.images[i], [28, 28]))
    ax[1, i].imshow(np.reshape(samples[i], [28, 28]))
    # ax[1, i].imshow(np.reshape(samples[i], [28, 28]), cmap='gray')

plt.tight_layout()
plt.show()
