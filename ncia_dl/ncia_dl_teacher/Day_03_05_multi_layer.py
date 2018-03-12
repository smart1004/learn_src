# Day_03_05_multi_layer.py
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

def show_accuracy(hypothesis, sess, x, y, keep_rate, prompt, dataset):
    pred = tf.equal(tf.argmax(hypothesis, axis=1),
                    tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    # print('-' * 50, 'show_accuracy')
    print(prompt, sess.run(accuracy, {x: dataset.images, y: dataset.labels, keep_rate: 1.0}))

def softmax(x, y, keep_rate=None):
    # 784 : feature
    #  10 : class
    w = tf.Variable(tf.zeros([784, 10]))  # 784 = 28 * 28 * 1
    b = tf.Variable(tf.zeros([10]))

    z = tf.matmul(x, w) + b
    hypothesis = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
    cost = tf.reduce_mean(cost_i)

    return hypothesis, cost


def multi_layer_relu(x, y, keep_rate=None):
    w1 = tf.Variable(tf.random_normal([784, 256]))
    w2 = tf.Variable(tf.random_normal([256, 256]))
    w3 = tf.Variable(tf.random_normal([256, 10]))

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    # hypothesis = z
    z = tf.matmul(r2, w3) + b3

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
    cost = tf.reduce_mean(cost_i)

    return z, cost


def multi_layer_xavier(x, y, keep_rate=None):
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

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    # hypothesis = z
    z = tf.matmul(r2, w3) + b3

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
    cost = tf.reduce_mean(cost_i)

    return z, cost


def multi_layer_dropout(x, y, keep_rate=None):
    w1 = tf.get_variable('w1_', shape=[784, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2_', shape=[256, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3_', shape=[256, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    w4 = tf.get_variable('w4_', shape=[256, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    w5 = tf.get_variable('w5_', shape=[256, 10],
                         initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([256]))
    b4 = tf.Variable(tf.zeros([256]))
    b5 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_rate)

    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_rate)

    z3 = tf.matmul(d2, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3, keep_rate)

    z4 = tf.matmul(d3, w4) + b4
    r4 = tf.nn.relu(z4)
    d4 = tf.nn.dropout(r4, keep_rate)

    # hypothesis = z
    z = tf.matmul(d4, w5) + b5

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
    cost = tf.reduce_mean(cost_i)

    return z, cost

def show_model(model):
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    keep_rate = tf.placeholder(tf.float32)

    # ---------------------------------- #
    hypothesis, cost = model(x, y, keep_rate)
    # ---------------------------------- #

    # optimizer = tf.train.GradientDescentOptimizer(0.01)
    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs, batch_size = 15, 128

    loops = mnist.train.num_examples // batch_size  # mnist.train.num_examples=55000,  batch_size=128
    # print('loops :', loops)  #loops:  429

    for i in range(epochs):
        total = 0
        for _ in range(loops):
            xx, yy = mnist.train.next_batch(batch_size)

            _, loss = sess.run([train, cost], feed_dict={x: xx, y: yy, keep_rate: 0.7})
            total += loss

        print('{:2} : {}'.format(i, total / loops))

    # --------------------------------- #
    show_accuracy(hypothesis, sess, x, y, keep_rate, 'train :', mnist.train)
    show_accuracy(hypothesis, sess, x, y, keep_rate, 'valid :', mnist.validation)
    show_accuracy(hypothesis, sess, x, y, keep_rate, 'test  :', mnist.test)
    sess.close()


show_model(softmax)
show_model(multi_layer_relu)
# show_model(multi_layer_xavier)
# show_model(multi_layer_dropout)

# [1] softmax
# train : 0.8978364
# valid : 0.904
# test  : 0.9055

# [2] multi_layer_relu
# train : 0.97474545
# valid : 0.9316
# test  : 0.9258

# [3] multi_layer_xavier
# train : 0.94429094
# valid : 0.9478
# test  : 0.9442

# [4] multi_layer_dropout
# train : 0.95336366
# valid : 0.9562
# test  : 0.9515

# adam.
# train : 0.996
# valid : 0.9812
# test  : 0.9811

# cnn, rnn, 강화
# slim, layers, tflearn
