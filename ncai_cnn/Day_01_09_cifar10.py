# Day_01_09_cifar10.py
import numpy as np
import tensorflow as tf
import Day_01_08_cifar10_load_images


# 문제
# 미니 데이터셋을 softmax와 cnn 코드에 연동해보세요.

def softmax():
    data = Day_01_08_cifar10_load_images.read_dataset('cifar10_mini')
    labels, images = data

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.int32)

    w = tf.Variable(tf.zeros([32 * 32 * 3, 3]))
    b = tf.Variable(tf.zeros([3]))

    z = tf.matmul(x, w) + b
    hypothesis = tf.nn.softmax(z)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                     logits=z)
    cost = tf.reduce_mean(cost_i)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={x: images,
                                   y: labels})

        if i % 20 == 0:
            print(i, sess.run(cost, feed_dict={x: images,
                                               y: labels}))

    prediction = tf.equal(tf.argmax(hypothesis, 1),
                          tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: images}))

    sess.close()


def cnn():
    data = Day_01_08_cifar10_load_images.read_dataset('cifar10_mini')
    labels, images = data
    print(images.shape, labels.shape)
    # (30, 3072) (30, 3)

    x = tf.placeholder(tf.float32, [None, 3072])
    y = tf.placeholder(tf.int32, [None, 3])
    keep_rate = tf.placeholder(tf.float32)

    # ---------------------------------- #

    w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1))
    w_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    w_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 256], stddev=0.1))
    w_fc2 = tf.Variable(tf.truncated_normal([256, 3], stddev=0.1))

    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[3]))

    # ---------------------------------- #

    x_image = tf.reshape(x, [-1, 32, 32, 3])

    # (?, 32, 32, 32)
    c_conv1 = tf.nn.conv2d(x_image, w_conv1,
                           strides=[1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(c_conv1 + b_conv1)
    # (?, 16, 16, 32)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # (?, 16, 16, 64)
    c_conv2 = tf.nn.conv2d(h_pool1, w_conv2,
                           strides=[1, 1, 1, 1], padding='SAME')
    h_conv2 = tf.nn.relu(c_conv2 + b_conv2)
    # (?, 8, 8, 64)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # 컨볼루션 --> 풀리 커넥티드
    h_flats = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_flats, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_rate)

    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    # hypothesis = tf.nn.softmax(y_conv)
    hypothesis = y_conv

    # -------------------------- #

    cost_i = tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=y_conv)
    cost = tf.reduce_mean(cost_i)
    train = tf.train.AdamOptimizer(0.001).minimize(cost)

    # ---------------------------------- #

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs, batch_size = 1, 100
    count = 5
    for i in range(epochs):
        for _ in range(count):
            sess.run(train, feed_dict={x: images,
                                       y: labels,
                                       keep_rate: 0.5})
        print('epochs :', i)

    sess.close()


# softmax()
cnn()


