# Day_01_05_MultiLinear.py
import tensorflow as tf


def multi_1():
    # y = w1 * x1 + w2 * x2 + b
    #      1         1        0
    # y = x1 + x2
    x1 = [1, 0, 3, 0, 5]
    x2 = [0, 2, 0, 4, 0]
    y  = [1, 2, 3, 4, 5]

    w1 = tf.Variable(tf.random_uniform([1], -1, 1))
    w2 = tf.Variable(tf.random_uniform([1], -1, 1))
    b  = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w1 * x1 + w2 * x2 + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# x1과 x2를 하나의 변수로 합쳐주세요. (tf.matmul)
def multi_2():
    x = [[1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    #
    w = tf.Variable(tf.random_uniform([?, ?], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    # hypothesis = w[0] * x[0] + w[1] * x[1] + b

    # (1, 5) = (?, ?) x (2, 5)
    hypothesis = tf.matmul(w, x) + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# bias를 없애보세요.
def multi_3():
    # y = w1 * x1 + w2 * x2 + b
    # x = [[1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 0.],
    #      [1., 1., 1., 1., 1.]]

    # y = w1 * x1 + b + w2 * x2
    # x = [[1., 0., 3., 0., 5.],
    #      [1., 1., 1., 1., 1.],
    #      [0., 2., 0., 4., 0.]]

    # y = b + w1 * x1 + w2 * x2
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([?, ?], -1, 1))

    # hypothesis = w[0] * x[0] + w[1] * x[1] + w[2] * x[?]
    # hypothesis = w[0] * x[0] + w[1] * x[1] +  b   * x[?]

    # (1, 5) = (?, ?) x (3, 5)
    hypothesis = tf.matmul(w, x)
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        # print(i, sess.run(cost))
        print(i, sess.run(w))

    sess.close()


# 문제
# placeholder 버전으로 수정해서
# (5, 7)일 때와 (5, 12)일 때의 값을 예측해보세요.
def multi_4():
    xx = [[1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.],
          [1., 1., 1., 1., 1.]]
    y = [1, 2, 3, 4, 5]

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(w, x)
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

    # print(sess.run(hypothesis, {x: [5, 7]}))
    # print(sess.run(hypothesis, {x: [5, 7, 1]}))
    print(sess.run(hypothesis, {x: xx}))
    print(sess.run(hypothesis, {x: [[1., 0., 3., 0., 5.],
                                    [0., 2., 0., 4., 0.],
                                    [1., 1., 1., 1., 1.]]}))
    print(sess.run(hypothesis, {x: [[5., 5.],
                                    [7., 12.],
                                    [1., 1.]]}))

    sess.close()


# 문제
# 행렬 곱셈에서 w와 x의 위치를 바꿔주세요.
# (5, 7)일 때와 (5, 12)일 때의 값을 예측해보세요.
def multi_5():
    xx = [[1., 1., 0.],
          [1., 0., 2.],
          [1., 3., 0.],
          [1., 0., 4.],
          [1., 5., 0.]]
    y = [[1], [2], [3], [4], [5]]  # (5, 1)

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([?, ?], -1, 1))

    # (5, 1) = (5, 3) x (3, 1)
    hypothesis = tf.matmul(x, w)
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

    print(sess.run(hypothesis, {x: xx}))
    print(sess.run(hypothesis, {x: [[1., 5., 7.],
                                    [1., 5., 12.]]}))
    sess.close()


multi_5()

