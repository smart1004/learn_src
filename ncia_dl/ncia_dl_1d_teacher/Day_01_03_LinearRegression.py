# Day_01_03_LinearRegression.py
import tensorflow as tf


# 문제
# x가 5와 7일 때의 y값을 예측해보세요.
def regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    # hypothesis = tf.add(tf.multiply(w, x), b)
    hypothesis = w * x + b  #shape=(3,) x가 3열이라서

    # cost = tf.reduce_mean(tf.square(hypothesis - y))
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    ww = sess.run(w)
    bb = sess.run(b)
    print(ww, bb)
    print('5 :', ww * 5 + bb)
    print('7 :', ww * 7 + bb)

    sess.close()
# regression_1()

# 문제
# 위의 코드를 placeholder 버전으로 변환하세요.
def regression_2():
    xx = [1, 2, 3]
    yy = [1, 2, 3]

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w * x + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={x: xx, y: yy})
        print(i, sess.run(cost, {x: xx, y: yy}))

    # 문제
    # x가 5와 7일 때의 y값을 예측해보세요.
    print(sess.run(hypothesis, {x: 5}))
    print(sess.run(hypothesis, {x: 7}))

    sess.close()

# regression_2()

# 문제
# 위의 코드에서 부족한 부분 2가지를 찾으세요.
def regression_3():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w * x + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={x: xx})
        print(i, sess.run(cost, {x: xx}))

    # 문제
    # x가 5와 7일 때의 y값을 예측해보세요.
    print(sess.run(hypothesis, {x: xx}))
    print(sess.run(hypothesis, {x: [1, 2, 3]}))
    print(sess.run(hypothesis, {x: [5, 7]}))

    sess.close()
regression_3()
