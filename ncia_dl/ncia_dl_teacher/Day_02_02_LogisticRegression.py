# Day_02_02_LogisticRegression.py
import tensorflow as tf
import numpy as np


def logistic_1():
    x = [[1., 1., 1., 1., 1., 1.],
         [2., 3., 3., 5., 7., 2.],
         [1., 2., 5., 5., 5., 5.]]
    y = [0, 0, 0, 1, 1, 1]
    y = np.array(y)

    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    z = tf.matmul(w, x)
    hypothesis = tf.nn.sigmoid(z)
    # hypothesis = 1 / (1 + tf.exp(-z))

    cost = tf.reduce_mean(   y  * -tf.log(  hypothesis) +
                          (1-y) * -tf.log(1-hypothesis))
    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# 3시간 공부하고 8번 출석했을 때와
# 7시간 공부하고 2번 출석했을 때의 결과를 알려주세요.
def logistic_2():
    xx = [[1., 1., 1., 1., 1., 1.],
          [2., 3., 3., 5., 7., 2.],
          [1., 2., 5., 5., 5., 5.]]
    y = [0, 0, 0, 1, 1, 1]
    y = np.array(y)

    # x = tf.placeholder(tf.float32)
    # x = tf.placeholder(tf.float32, shape=[3, 6])      # error.
    x = tf.placeholder(tf.float32, shape=[3, None])
    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    # (1, 2) = (1, 3) x (3, 2)
    z = tf.matmul(w, x)
    hypothesis = tf.nn.sigmoid(z)
    # hypothesis = 1 / (1 + tf.exp(-z))

    cost = tf.reduce_mean(   y  * -tf.log(  hypothesis) +
                          (1-y) * -tf.log(1-hypothesis))
    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: xx})

        if i % 20 == 0:
            print(i, sess.run(cost, {x: xx}))

    y_hat = sess.run(hypothesis, {x: xx})
    print(y_hat)

    y_hat = sess.run(hypothesis, {x: [[1., 1.],
                                      [3., 7.],
                                      [8., 2.]]})
    print(y_hat)
    print(y_hat >= 0.5)
    print(y_hat.shape)
    print(sess.run(w))
    # [[-4.5166197   0.39605284  0.7939856 ]]

    sess.close()




